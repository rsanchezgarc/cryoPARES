import os.path as osp
from starstack.particlesStar import ParticlesStarSet
from typing import Union, Optional
from os import PathLike
from cryoPARES.datamanager.particlesDataset import ParticlesDataset

class ParticlesRelionStarDataset(ParticlesDataset):
    """
    ParticlesRelionStarDataset: A Pytorch Dataset for dealing with Cryo-EM particles in RELION star format.<br>
    It can download data automatically

    ```python
    ds = ParticlesRelionStarDataset(starFname="/tmp/cryoSupervisedDataset/particles.star", rootDir="/tmp/cryoSupervisedDataset/", symmetry="c1)
    ```
    <br>
    """

    def __init__(self, starFname: Union[PathLike, str], rootDir: Optional[str], symmetry: str, **kwargs):
        """
        ##Builder

        Args:
            starFname (Union[PathLike, str]): The star filename to use
            rootDir (str): The root directory where the stack files are
            symmetry (str): The point symmetry of the macromolecule
        """

        super().__init__(symmetry=symmetry, **kwargs)
        self._starFname = str(starFname)
        self._datadir = osp.expanduser(rootDir) if rootDir is not None else None

    def load_ParticlesStarSet(self):
        return ParticlesStarSet(starFname=self._starFname, particlesDir=self._datadir)


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
    parts = ParticlesRelionStarDataset(starFname=osp.expanduser(args.filename),
                                       rootDir=args.dirname,
                                       symmetry=args.symmetry,
                                       desired_image_size_px=args.resize_box,
                                       desired_sampling_rate_angs=args.sampling_rate,
                                       mask_radius_angs=args.mask_radius_angs,
                                       perImg_normalization=args.normalization_type,
                                       ctf_correction=args.ctf_correction
                                       )

    print(len(parts))
    from matplotlib import pyplot as plt
    channels_to_show = args.channels_to_show
    for elem in parts:
        iid, img, *_ = elem
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