import os
import re
import json
import click
import torch
import dnnlib_amed
from torch_utils import distributed as dist
from training_amed import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')

#----------------------------------------------------------------------------

@click.command()

# General options.
@click.option('--dataset_name',     help='Dataset name for inpainting (e.g., celeba-hq, imagenet)', metavar='STR', required=True)
@click.option('--outdir',           help='Where to save the results', metavar='DIR', type=str, default='./exps')
@click.option('--total_kimg',       help='Number of images (k) for training', metavar='INT', type=int, default=10)
@click.option('--model_path_64',    help='Path to pre-trained LWDM 64x64 checkpoint', metavar='PATH', type=str, required=True)
@click.option('--model_path_256',   help='Path to pre-trained LWDM 256x256 checkpoint', metavar='PATH', type=str, required=True)
@click.option('--data_dir',         help='Path to training dataset (images and masks)', metavar='DIR', type=str, required=True)

# Options for solvers
@click.option('--num_steps',        help='Number of time steps for training', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--sampler_stu',      help='Student solver', metavar='STR', type=click.Choice(['amed', 'dpm', 'dpmpp', 'euler', 'ipndm']), default='amed', show_default=True)
@click.option('--sampler_tea',      help='Teacher solver', metavar='STR', type=click.Choice(['heun', 'dpm', 'dpmpp', 'euler', 'ipndm']), default='heun', show_default=True)
@click.option('--M',                help='Steps to insert between two adjacent steps', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--guidance_type',    help='Guidance type', type=click.Choice(['cg', 'cfg', 'uncond', None]), default=None, show_default=True)
@click.option('--guidance_rate',    help='Guidance rate', metavar='FLOAT', type=float, default=0.)
@click.option('--schedule_type',    help='Time discretization schedule', metavar='STR', type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='discrete', show_default=True)
@click.option('--schedule_rho',     help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--afs',              help='Whether to use AFS', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--scale_dir',        help='Scale the gradient by [1-scale_dir, 1+scale_dir]', metavar='FLOAT', type=click.FloatRange(min=0), default=0.01, show_default=True)
@click.option('--scale_time',       help='Scale the gradient by [1-scale_time, 1+scale_time]', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--max_order',        help='Max order for solvers', metavar='INT', type=click.IntRange(min=1), default=3)
@click.option('--predict_x0',       help='Whether to use data prediction mode', metavar='BOOL', type=bool, default=True)
@click.option('--lower_order_final', help='Lower the order at final stages', metavar='BOOL', type=bool, default=True)

# Hyperparameters.
@click.option('--batch',            help='Total batch size', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1))
@click.option('--lr',               help='Learning rate', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=5e-3, show_default=True)

# Performance-related.
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL', type=bool, default=True, show_default=True)

# I/O-related.
@click.option('--desc',             help='String to include in result dir name', metavar='STR', type=str)
@click.option('--nosubdir',         help='Do not create a subdirectory for results', is_flag=True)
@click.option('--tick',             help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--snap',             help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dump',             help='How often to dump state', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',             help='Random seed [default: random]', metavar='INT', type=int)
@click.option('-n', '--dry-run',    help='Print training options and exit', is_flag=True)
@click.option('--num_channels', help='Base number of channels in UNet', type=int, default=256)
@click.option('--channel_mult', help='Channel multipliers (comma-separated)', type=str, default="1,1,2,2,4,4")
@click.option('--num_res_blocks', help='Number of residual blocks per resolution', type=int, default=2)
@click.option('--attention_resolutions', help='Resolutions with attention (comma-separated)', type=str, default="32,16,8")

def main(**kwargs):
    opts = dnnlib_amed.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    #change:Convert channel_mult and attention_resolutions to tuples
    opts.channel_mult = tuple(map(int, opts.channel_mult.split(',')))
    opts.attention_resolutions = tuple(map(int, opts.attention_resolutions.split(',')))

    # Initialize config dict for both resolutions.
    c_64 = dnnlib_amed.EasyDict()
    c_256 = dnnlib_amed.EasyDict()
    for c in [c_64, c_256]:
        # change:
        c.update(num_channels=opts.num_channels, channel_mult=opts.channel_mult,num_res_blocks=opts.num_res_blocks, attention_resolutions=opts.attention_resolutions)

        c.loss_kwargs = dnnlib_amed.EasyDict()
        c.AMED_kwargs = dnnlib_amed.EasyDict()
        c.optimizer_kwargs = dnnlib_amed.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)

    # AMED predictor architecture for 64x64 and 256x256.
    for c in [c_64, c_256]:
        c.AMED_kwargs.class_name = 'training.networks.AMED_predictor'
        c.AMED_kwargs.update(num_steps=opts.num_steps, sampler_stu=opts.sampler_stu, sampler_tea=opts.sampler_tea,
                             M=opts.m, guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate,
                             schedule_rho=opts.schedule_rho, schedule_type=opts.schedule_type, afs=opts.afs,
                             dataset_name=opts.dataset_name, scale_dir=opts.scale_dir, scale_time=opts.scale_time,
                             max_order=opts.max_order, predict_x0=opts.predict_x0, lower_order_final=opts.lower_order_final)
        c.loss_kwargs.class_name = 'training.loss.AMED_loss'

    # Training options.
    for c in [c_64, c_256]:
        c.total_kimg = opts.total_kimg
        c.kimg_per_tick = 1
        c.snapshot_ticks = c.total_kimg
        c.state_dump_ticks = c.total_kimg
        c.update(dataset_name=opts.dataset_name, batch_size=opts.batch, batch_gpu=opts.batch_gpu, gpus=dist.get_world_size(), cudnn_benchmark=opts.bench)
        c.update(guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate, data_dir=opts.data_dir)

    # Random seed.
    if opts.seed is not None:
        c_64.seed = opts.seed
        c_256.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c_64.seed = int(seed)
        c_256.seed = int(seed)

    # Description string and output directory for both resolutions.
    nfe = 2 * (opts.num_steps - 1) - 1 if opts.afs else 2 * (opts.num_steps - 1)
    desc_base = f'{opts.dataset_name}-{opts.num_steps}-{nfe}-{opts.sampler_stu}-{opts.sampler_tea}-{opts.m}-{opts.schedule_type}{opts.schedule_rho}'
    desc = desc_base + '-afs' if opts.afs else desc_base
    if opts.desc:
        desc += f'-{opts.desc}'
    for c, res in [(c_64, '64'), (c_256, '256')]:
        c.run_dir = os.path.join(opts.outdir, f'{res}/{desc}') if not opts.nosubdir else opts.outdir
        if dist.get_rank() == 0 and not opts.nosubdir:
            prev_run_dirs = [x for x in os.listdir(os.path.join(opts.outdir, res)) if os.path.isdir(os.path.join(opts.outdir, res, x))] if os.path.isdir(os.path.join(opts.outdir, res)) else []
            prev_run_ids = [int(re.match(r'^\d+', x).group()) for x in prev_run_dirs if re.match(r'^\d+', x)]
            cur_run_id = max(prev_run_ids, default=-1) + 1
            c.run_dir = os.path.join(opts.outdir, res, f'{cur_run_id:05d}-{desc}')
            assert not os.path.exists(c.run_dir)

    # Pass LWDM paths.
    c_64.model_path = opts.model_path_64
    c_256.model_path = opts.model_path_256

    # Print options.
    dist.print0('Training options for 64x64:')
    dist.print0(json.dumps(c_64, indent=2))
    dist.print0(f'Output directory: {c_64.run_dir}')
    dist.print0('Training options for 256x256:')
    dist.print0(json.dumps(c_256, indent=2))
    dist.print0(f'Output directory: {c_256.run_dir}')

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directories and train.
    for c, res in [(c_64, '64x64'), (c_256, '256x256')]:
        if dist.get_rank() == 0:
            os.makedirs(c.run_dir, exist_ok=True)
            with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
                json.dump(c, f, indent=2)
            dnnlib_amed.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
        dist.print0(f'Training AMED predictor for {res}...')
        training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()