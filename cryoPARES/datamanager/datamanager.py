import torch

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME

def get_number_image_channels():
     if main_config.datamanager.ctf_correction_mode.startswith("concat"):
         return 2
     else:
         return 1

def get_example_random_batch(batch_size=1):
    #TODO: batch size should be in the config
    imgsize = main_config.datamanager.desired_image_size_pixels
    batch = {BATCH_PARTICLES_NAME:torch.randn(batch_size, get_number_image_channels(), imgsize, imgsize)}
    return batch