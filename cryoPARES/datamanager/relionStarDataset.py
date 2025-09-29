import os.path as osp
import subprocess

import pandas as pd
from starstack.particlesStar import ParticlesStarSet, split_particle_and_fname
from typing import Union, Optional, List, Tuple
from os import PathLike

from cryoPARES.constants import BATCH_PARTICLES_NAME
from cryoPARES.datamanager.particlesDataset import ParticlesDataset

class ParticlesRelionStarDataset(ParticlesDataset):
    """
    ParticlesRelionStarDataset: A Pytorch Dataset for dealing with Cryo-EM particles in RELION star format.<br>
    It can download data automatically

    ```python
    ds = ParticlesRelionStarDataset(star_fname="/tmp/cryoSupervisedDataset/particles.star", particles_dir="/tmp/cryoSupervisedDataset/", symmetry="c1)
    ```
    <br>
    """

    def __init__(self, particles_star_fname: Union[PathLike, str], particles_dir: Optional[str],
                 subset_idxs: Optional[List[int]] = None
                 , **kwargs):
        """
        ##Builder

        Args:
            particles_star_fname (Union[PathLike, str]): The star filename to use
            particles_dir (str): The root directory where the stack files are
            subset_idxs (Optional[List[int]] ): The subset of idxs to use
        """

        super().__init__(subset_idxs=subset_idxs, **kwargs)
        self._star_fname = particles_star_fname
        if not isinstance(particles_star_fname, dict):
            self._star_fname = str(particles_star_fname)
        self._datadir = osp.expanduser(particles_dir) if particles_dir is not None else None

    def load_ParticlesStarSet(self):
        ps = ParticlesStarSet(starFname=self._star_fname, particlesDir=self._datadir)

        ulimit = subprocess.run(["ulimit -n"], check=True, capture_output=True, shell=True)
        assert ulimit.returncode == 0, "Error, ulimit -n command failed"
        ulimit = int(ulimit.stdout.decode().strip())
        stackFnames = ps.particles_md["rlnImageName"].map(lambda x: split_particle_and_fname(x)["basename"])
        n_stacks = len(stackFnames.unique())
        assert n_stacks < 5 * ulimit, f"Error, the number of particle stacks is too" \
                                      f" large ({n_stacks}) given current number of allowed file " \
                                      f"handlers {ulimit}. Please, increase the number using " \
                                      f"'ulimit -n NUMBER', where NUMBER should be much larger than " \
                                      f"the number of stacks (at least 10x). Otherwise, group the " \
                                      f"particle stacks (.mrcs) into less but larger stack files."
        return ps


    def saveMd(self, fname: Union[str, PathLike], overwrite: bool = True):
        """
        Saves the metadata of the current PartcilesDataset as a starfile
        Args:
            fname: The name of the file were the metadata will be saved
            overwrite: If true, overwrites the file fname if already exists

        """
        assert fname.endswith(".star"), "Error, metadata files will be saved as star files. Change extension to .star"
        self.particles.save(starFname=fname, overwrite=overwrite)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Visualize dataset relion starDataset")
    parser.add_argument("-f", "--filename", type=str, help="The starfile to visualize", required=True)
    parser.add_argument("-d", "--dirname", type=str, help="The root directory for the particle stacks", required=False,
                        default=None)
    parser.add_argument("-s", "--symmetry", type=str, help="The point symmetry of the particle", required=True,
                        default=None)
    parser.add_argument("-r", "--sampling_rate", type=float, help="The desired sampling rate", required=False,
                        default=None)
    parser.add_argument("-b", "--resize_box", type=int, help="The desired image box size", required=False, default=None)
    parser.add_argument("-n", "--normalization_type", help="The normalization type",
                        choices=["none", "noiseStats", "subtractMean"], default="noiseStats", required=False)
    parser.add_argument("-c", "--ctf_correction", type=str, help="The ctf correction type",
                        choices=["none", "phase_flip", "concat_phase_flip"], default="none", required=False)
    parser.add_argument( "--channels_to_show", type=int, nargs="+", default=[0], required=False)
    parser.add_argument( "--mask_radius_angs", type=float,  default=None, required=False)

    args = parser.parse_args()
    kwargs = {}
    if args.resize_box:
        kwargs["image_size_px_for_nnet"] = args.resize_box
    if args.resize_box:
        kwargs["sampling_rate_angs_for_nnet"] = args.sampling_rate
    if args.mask_radius_angs:
        kwargs["mask_radius_angs"] = args.mask_radius_angs
    if args.ctf_correction:
        kwargs["ctf_correction"] = args.ctf_correction
    parts = ParticlesRelionStarDataset(particles_star_fname=osp.expanduser(args.filename),
                                       particles_dir=args.dirname,
                                       symmetry=args.symmetry, halfset=None, min_maxProb=None,
                                       store_data_in_memory=True,
                                       **kwargs)

    print(len(parts))
    from matplotlib import pyplot as plt
    channels_to_show = args.channels_to_show
    for elem in parts:
        img = elem[BATCH_PARTICLES_NAME]
        print(img.shape)
        # continue
        assert 1 <= len(channels_to_show) <= 4, "Error, at least one channel required and no more than 4"
        f, axes = plt.subplots(1, len(channels_to_show), squeeze=False)
        for j, c in enumerate(channels_to_show):
            axes[0, c].imshow(img[c, ...], cmap="gray")
        plt.show()
        plt.close()
    print("Done")


"""
python -m  cryoPARES.datamanager.relionStarDataset -f ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/1000proj.star  -s c1
"""