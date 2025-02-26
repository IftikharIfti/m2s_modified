"""Main training loop."""

import os
import csv
import time
import copy
import json
import pickle
import numpy as np
import torch
import dnnlib_amed
import random
from torch import autocast
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from guided_diffusion.image_datasets import load_data

#----------------------------------------------------------------------------

def load_lwdm_model(model_path, image_size, device, num_channels, channel_mult, num_res_blocks, attention_resolutions):
    """
    Load the pre-trained LWDM model from a .pt checkpoint with M2S-specific UNet configuration.
    
    Args:
        model_path (str): Path to the LWDM checkpoint (.pt file).
        image_size (int): Image resolution (64 or 256).
        device (torch.device): Device to load the model onto.
        num_channels (int): Base number of channels in UNet.
        channel_mult (str): Channel multipliers as a comma-separated string (e.g., "1,1,2,2,4,4").
        num_res_blocks (int): Number of residual blocks per resolution.
        attention_resolutions (tuple): Resolutions where attention is applied.
    
    Returns:
        tuple: (model, diffusion) - Loaded UNetModel and GaussianDiffusion objects.
    """
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
    args = dnnlib_amed.EasyDict(model_and_diffusion_defaults())
    # Configure UNet to match M2S LWDM
    args.image_size = image_size
    args.num_channels = num_channels
    args.channel_mult = channel_mult  # Keep as string
    args.num_res_blocks = num_res_blocks
    args.attention_resolutions = attention_resolutions
    model, diffusion = create_model_and_diffusion(**args)
    checkpoint = torch.load(model_path, map_location="cpu")
    # Assuming direct state_dict from M2S's TrainLoop.save()
    model.load_state_dict(checkpoint)
    model.to(device).eval()
    # AMED-specific attributes
    model.sigma_min = 0.002
    model.sigma_max = 80.0
    model.img_resolution = image_size
    model.img_channels = 3
    return model, diffusion

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    AMED_kwargs         = {},       # Options for AMED predictor.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = None,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 20,       # Training duration, measured in thousands of training images.
    kimg_per_tick       = 1,        # Interval of progress prints.
    snapshot_ticks      = 1,        # How often to save network snapshots, None = disable.
    state_dump_ticks    = 20,       # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    dataset_name        = None,     # Dataset name (e.g., "celeba-hq").
    data_dir            = None,     # Path to dataset (images and masks).
    model_path          = None,     # Path to LWDM checkpoint.
    guidance_type       = None,     # Guidance type (unused in AMED).
    guidance_rate       = 0.,       # Guidance rate (unused in AMED).
    num_channels        = 256,      # Base number of channels in UNet.
    channel_mult        = "1,1,2,2,4,4",  # Channel multipliers as string.
    num_res_blocks      = 2,        # Number of residual blocks per resolution.
    attention_resolutions = (32, 16, 8),  # Resolutions with attention.
    device              = torch.device('cuda'),
    **kwargs,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load pre-trained LWDM model.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    
    image_size = 64 if '64' in model_path else 256
    net, diffusion = load_lwdm_model(
        model_path, 
        image_size=image_size, 
        device=device,
        num_channels=num_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions
    )
    
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Load dataset with images and masks.
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_gpu,
        image_size=image_size,
        class_cond=False,
        deterministic=False,
        random_flip=False,
    )
    
    # Construct AMED predictor.
    dist.print0('Constructing AMED predictor...')
    AMED_kwargs.update(img_resolution=net.img_resolution)
    AMED_predictor = dnnlib_amed.util.construct_class_by_name(**AMED_kwargs)
    AMED_predictor.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_kwargs.update(
        num_steps=AMED_kwargs.num_steps, 
        sampler_stu=AMED_kwargs.sampler_stu, 
        sampler_tea=AMED_kwargs.sampler_tea,
        M=AMED_kwargs.M, 
        schedule_type=AMED_kwargs.schedule_type, 
        schedule_rho=AMED_kwargs.schedule_rho,
        afs=AMED_kwargs.afs, 
        max_order=AMED_kwargs.max_order, 
        sigma_min=net.sigma_min, 
        sigma_max=net.sigma_max,
        predict_x0=AMED_kwargs.predict_x0, 
        lower_order_final=AMED_kwargs.lower_order_final
    )
    loss_fn = dnnlib_amed.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib_amed.util.construct_class_by_name(params=AMED_predictor.parameters(), **optimizer_kwargs)
    ddp = torch.nn.parallel.DistributedDataParallel(AMED_predictor, device_ids=[device], broadcast_buffers=False)

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    data_iterator = iter(data)
    while True:
        # Load batch with images and masks.
        try:
            batch_data = next(data_iterator)
            images, masks = batch_data[0].to(device), batch_data[1].to(device)  # Assuming masks are second item
        except StopIteration:
            data_iterator = iter(data)
            batch_data = next(data_iterator)
            images, masks = batch_data[0].to(device), batch_data[1].to(device)

        # Generate latents and conditions.
        latents = loss_fn.sigma_max * torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        model_kwargs = {'ref_img': images}
        model_mask_kwargs = {'ref_img': masks}

        # Generate teacher trajectories with inpainting.
        with torch.no_grad():
            teacher_traj = loss_fn.get_teacher_traj(net=net, tensor_in=latents, model_kwargs=model_kwargs, model_mask_kwargs=model_mask_kwargs)

        # Perform training step by step.
        for step_idx in range(loss_fn.num_steps - 1):
            optimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    loss, stu_out = loss_fn(
                        AMED_predictor=ddp, 
                        net=net, 
                        tensor_in=latents, 
                        step_idx=step_idx, 
                        teacher_out=teacher_traj[step_idx],
                        model_kwargs=model_kwargs, 
                        model_mask_kwargs=model_mask_kwargs
                    )
                    training_stats.report('Loss/loss', loss)
                    loss.sum().mul(1 / batch_gpu_total).backward()

            for param in AMED_predictor.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer.step()
            
            if AMED_predictor.sampler_stu in ['euler', 'dpm', 'amed']:
                latents = teacher_traj[step_idx]
            else:
                latents = stu_out

        # Maintenance tasks.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = [
            f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}",
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}",
            f"time {dnnlib_amed.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        dist.print0(' '.join(fields))

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick > 0:
            data = dict(model=AMED_predictor, loss_fn=loss_fn)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data

        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    dist.print0('Exiting...')