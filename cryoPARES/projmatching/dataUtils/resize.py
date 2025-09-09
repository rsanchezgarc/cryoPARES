from typing import Tuple

import skimage.transform
import torch.nn.functional as tF
from joblib import delayed, Parallel
from starstack import ParticlesStarSet
from torchCryoAlign.converts import get_vol, write_vol
from torchCryoAlign.dataUtils.dataTypes import PARTICLES_SET_OR_STARFNAME
from torchCryoAlign.myProgressBar import myTrange


def _get_new_size(shape:Tuple[int],  current_sampling_rate, downsampling_factor:float):
    new_size = [ int(s/downsampling_factor) if s>1 else 1 for s in shape] #TODO: Check that the map is a cube
    new_size = [s + 1 if (s % 2) else s if s>1 else 1 for s in new_size]
    new_sampling_rate = current_sampling_rate * shape[-1]/new_size[-1]
    return new_size, new_sampling_rate
def resize_vol_fname(reference_vol:str, new_reference_vol:str, downsampling_factor:float):
    vol = get_vol(reference_vol, pixel_size=None)
    new_size, new_sampling_rate = _get_new_size(vol.shape, vol.pixel_size, downsampling_factor)
    vol = tF.interpolate(vol.unsqueeze(0).unsqueeze(0), size=new_size, mode="trilinear").squeeze(0).squeeze(0)
    vol.pixel_size = new_sampling_rate
    write_vol(vol, new_reference_vol)
    return new_size, new_sampling_rate


_STAR_SET = None #TODO: Should this be cleaned at some point?
def resize_particles(particles:PARTICLES_SET_OR_STARFNAME, data_rootdir, new_star_in_fname, downsampling_factor, n_jobs, verbose=False):

    if not isinstance(particles, ParticlesStarSet):
        starSet = ParticlesStarSet(particles, data_rootdir)
    else:
        starSet = particles.copy()
    optics_md = starSet.optics_md.copy()
    particles_md = starSet.particles_md.copy()
    new_size, new_sampling_rate = _get_new_size(starSet.particle_shape, starSet.sampling_rate, downsampling_factor)

    optics_md["rlnImageSize"] = new_size[-1]
    optics_md["rlnImagePixelSize"] = new_sampling_rate

    def _resize_img(i):
        global _STAR_SET
        if _STAR_SET is None:
            _STAR_SET = starSet
        img, md = _STAR_SET[i]
        img = skimage.transform.resize(img, output_shape=new_size)
        return img
    npImages = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(_resize_img)(i)
                                                              for i in myTrange(len(starSet),
                                                                              desc="Resizing particles",
                                                                              disable=not verbose,
                                                                              )
                                                              )
    parts = ParticlesStarSet.createFromPdNp(new_star_in_fname, optics_md, particles_md, npImages)
    parts.particles_md["beforeDownsamplingImageName"] = particles.particles_md['rlnImageName']
    return parts
