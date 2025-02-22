import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
# packages = [
#     "blobfile>=1.0.5", #not in amed
#     "torch",
#     "tqdm"
# ]
#REMINDER: pytoch_lightning and einops  and taming-transformers and transformers used inside ddpm of models_amed/ldm
packages = [
    "numpy",
    "pillow",
    "torchvision",
    "lpips",
    "torchmetrics",
    "click",
    "scipy",
    "psutil",
    "requests",
    "tqdm",
    "torch",
    "blobfile", 
    "imageio",
    "imageio-ffmpeg",
    "pyspng",
    "omegaconf",
    "pytorch_lightning",
    "einops",
    "taming-transformers",
    "transformers"
]
for package in packages:
    install(package)
