import os
import re
import os.path as osp
from typing import Optional, Literal, List
import numpy as np
import pandas as pd
import starfile


pattern = re.compile(r"(\d+@)?(.*/)?(.*)")


def split_particle_and_fname(fname):
    matchObj = re.match(pattern, fname)
    part_num = matchObj.group(1)[:-1] if matchObj.group(1) else None
    return dict(partNum=part_num, dirname=matchObj.group(2), basename=matchObj.group(3))


def modify_fnames(df: pd.DataFrame, col_name="rlnImageName", new_col_name=None,
                  dir_name=None, remove_idxs=False):
    if new_col_name is None:
        new_col_name = col_name
    if dir_name is None:
        dir_name = r"\g<2>"
    subst = dir_name + r"\g<3>" if remove_idxs else r"\g<1>" + dir_name + r"\g<3>"
    df[new_col_name] = df[col_name].map(lambda s: re.sub(pattern, subst, s))
    return df


def rebase_mdfname(df: pd.DataFrame, col_name="rlnImageName", dir_name="",
                   keep_basename_only=False, remove_zeros=True):
    if keep_basename_only:
        extract_fname = lambda chunks: (chunks["basename"],)
    else:
        extract_fname = lambda chunks: (chunks["dirname"], chunks["basename"]) \
            if chunks["dirname"] else (chunks["basename"],)

    def _process_fname(fname):
        chunks = split_particle_and_fname(fname)
        partnum = (str(int(chunks["partNum"])) if remove_zeros else chunks["partNum"]) + "@"
        return partnum + osp.join(dir_name, *extract_fname(chunks))

    df[col_name] = df[col_name].map(_process_fname)
    return df


def get_partNum_fname(fname, zero_based_requested=True, return_basename=False):
    part_num, image_name = fname.split("@")
    part_num = int(part_num)
    if zero_based_requested:
        part_num -= 1
    if return_basename:
        image_name = osp.basename(image_name)
    return part_num, image_name


def splitStarfile(star_fname: str, output_pattern: str, n_splits: int, shuffle: bool = True,
                  random_seed: Optional[int] = None, remove_stack_prefix: bool = True):
    """
    Split a particle starfile into several chunks.

    :param star_fname: Path to the input starfile
    :param output_pattern: Output filename prefix (chunk index and .star are appended)
    :param n_splits: Number of output chunks
    :param shuffle: Shuffle particles before splitting
    :param random_seed: Random seed for reproducible shuffling
    :param remove_stack_prefix: Remove directory prefix from particle stack filenames
    :return:
    """
    data = starfile.read(star_fname)
    optics = data["optics"]
    parts: pd.DataFrame = data["particles"]
    if remove_stack_prefix:
        parts = rebase_mdfname(parts, keep_basename_only=True)
    if shuffle:
        parts = parts.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    names = []
    for i, chunk_idx in enumerate(np.array_split(parts.index, n_splits)):
        new_name = output_pattern + "%d.star" % i
        starfile.write(dict(optics=optics, particles=parts.loc[chunk_idx]), new_name)
        names.append(new_name)
    return names


def rebaseStarfile(star_fname: str, out_star_fname: str, col_name: str = "rlnImageName",
                   dir_name: str = "", keep_basename_only: bool = True, remove_zeros: bool = True):
    """
    Change image filename paths in a starfile.

    :param star_fname: Path to the input starfile
    :param out_star_fname: Path to the output starfile
    :param col_name: Column containing filenames to modify
    :param dir_name: New directory prefix; new path will be {dir_name}/OLDNAME
    :param keep_basename_only: If true, OLDNAME is replaced by basename(OLDNAME)
    :param remove_zeros: Remove leading zeros from particle indices
    :return:
    """
    d = starfile.read(star_fname)
    if "particles" in d:
        df = d["particles"]
    else:
        df = d
    df = rebase_mdfname(df, col_name=col_name, dir_name=dir_name,
                        keep_basename_only=keep_basename_only, remove_zeros=remove_zeros)
    if "particles" in d:
        d["particles"] = df
    else:
        d = df
    starfile.write(d, out_star_fname, overwrite=True)


def selectSubset(star_fname: str, out_star_fname: str, select_value: str, col_name: str,
                 select_mode: Literal["gt", "lt", "ge", "le", "eq"] = "eq",
                 sort: bool = False):
    """
    Select a subset of particles based on a column value.

    :param star_fname: Path to the input starfile
    :param out_star_fname: Path to the output starfile
    :param select_value: Value to filter by (automatically cast to the column dtype)
    :param col_name: Column to filter on, e.g. "rlnCtfMaxResolution"
    :param select_mode: Comparison operator: eq, gt, lt, ge, le
    :param sort: Sort output rows by the filter column
    :return:
    """
    d = starfile.read(star_fname)
    if "particles" in d:
        df = d["particles"]
    else:
        df = d

    series = df[col_name]
    operation = getattr(series, f"__{select_mode}__")
    if series.dtype != bool:
        select_value = np.array(select_value).astype(series.dtype)
    else:
        select_value = np.array(select_value == "True")

    df = df[operation(select_value)]
    if sort:
        df.sort_values(col_name, inplace=True)
    if "particles" in d:
        d["particles"] = df
    else:
        d = df
    if out_star_fname:
        starfile.write(d, out_star_fname, overwrite=True)
    return d


def joinStars(star_fnames: List[str], out_star_fname: str):
    """
    Join several starfiles into one.

    :param star_fnames: List of input starfiles to join
    :param out_star_fname: Path to the output starfile
    :return:
    """
    optics = []
    particles = []
    for fname in star_fnames:
        d = starfile.read(fname)
        if "particles" in d:
            particles.append(d["particles"])
            optics.append(d["optics"])
        else:
            particles.append(d)

    particles = pd.concat(particles)

    if optics:
        optics = pd.concat(optics)
        optics.drop_duplicates(ignore_index=True, inplace=True)
        assert len(optics['rlnOpticsGroup']) == len(optics['rlnOpticsGroup'].unique()), \
            "Error, the starfiles have non-compatible optics groups"
        d = {"particles": particles, "optics": optics}
    else:
        d = particles
    if out_star_fname:
        starfile.write(d, out_star_fname, overwrite=True)
    return d


def removeNotFoundParticles(star_fname: str, out_star_fname: str, particles_dir: Optional[str] = None):
    """
    Remove particles whose stack files cannot be found on disk.

    :param star_fname: Path to the input starfile
    :param out_star_fname: Path to the output starfile
    :param particles_dir: Directory where stack files are located. Defaults to current working directory
    :return:
    """
    d = starfile.read(star_fname)
    if "particles" in d:
        df = d["particles"]
    else:
        df = d
    particles_dir = particles_dir if particles_dir else os.getcwd()
    found = df["rlnImageName"].map(
        lambda fname: osp.isfile(osp.join(particles_dir, get_partNum_fname(fname)[-1]))
    )
    print(f"{sum(found)}/{len(df)} particles found")
    df = df[found]
    if "particles" in d:
        d["particles"] = df
    else:
        d = df
    if out_star_fname:
        starfile.write(d, out_star_fname, overwrite=True)
    return d


def main():
    from argParseFromDoc import AutoArgumentParser, get_parser_from_function

    parser = AutoArgumentParser()
    subparsers = parser.add_subparsers(help='Subcommand', required=True, dest='command')

    p = subparsers.add_parser('splitStar', help='Split a particle starfile into several chunks')
    get_parser_from_function(splitStarfile, parser=p)
    p.set_defaults(func=splitStarfile)

    p = subparsers.add_parser('rebaseStar', help='Change image filenames to basenames or new directories')
    get_parser_from_function(rebaseStarfile, parser=p)
    p.set_defaults(func=rebaseStarfile)

    p = subparsers.add_parser('selectSubset', help='Select a subset of particles based on a column value')
    get_parser_from_function(selectSubset, parser=p)
    p.set_defaults(func=selectSubset)

    p = subparsers.add_parser('joinStars', help='Join several starfiles into one')
    get_parser_from_function(joinStars, parser=p)
    p.set_defaults(func=joinStars)

    p = subparsers.add_parser('removeNotFound', help='Remove particles whose stack files are not found on disk')
    get_parser_from_function(removeNotFoundParticles, parser=p)
    p.set_defaults(func=removeNotFoundParticles)

    arguments = parser.parse_args()
    func = arguments.func
    del arguments.func
    del arguments.command
    func(**vars(arguments))


if __name__ == "__main__":
    main()
