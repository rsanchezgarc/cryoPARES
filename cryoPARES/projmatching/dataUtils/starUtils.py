import os

import numba
import starfile
import numpy as np
import pandas as pd
from typing import Optional

from starstack import ParticlesStarSet
from cryoPARES.constants import RELION_ANGLES_NAMES, RELION_IMAGE_FNAME
from .dataTypes import PARTICLES_SET_OR_STARFNAME
from ..so3grid import SO3_discretizer

def split_halfsets(particles:PARTICLES_SET_OR_STARFNAME, starOutFnameTemplate:Optional[str]=None):
    """

    :param particles:
    :param starOutFnameTemplate:  E.g. "particles_%d.star
    :return:
    """
    particlesDf: pd.DataFrame
    if isinstance(particles, ParticlesStarSet):
        particlesDf = particles.particles_md
        opticsDf = particles.optics_md
    else:
        data = starfile.read(os.path.expanduser(particles))
        particlesDf= data["particles"]
        opticsDf = data["optics"]

    if "rlnRandomSubset" not in particlesDf:
        if "crossValidationGroupId" in particlesDf:
            subsetIdxs = particlesDf["crossValidationGroupId"] % 2
            subset1Idxs = np.where( subsetIdxs == 0)[0]
            subest2Idxs = np.where( subsetIdxs == 1)[0]
        else:
            n = len(particlesDf)
            all_indices = np.random.permutation(n)
            split_point = n // 2
            subset1Idxs = all_indices[:split_point]
            subest2Idxs = all_indices[split_point:]
        particlesDf["rlnRandomSubset"] = 0
        particlesDf["rlnRandomSubset"][subset1Idxs] = 1
        particlesDf["rlnRandomSubset"][subest2Idxs] = 2
    else:
        subset1Idxs = np.where(particlesDf["rlnRandomSubset"] == 1)[0]
        subest2Idxs = np.where(particlesDf["rlnRandomSubset"] == 2)[0]


    if starOutFnameTemplate is not None:
        starfile.write(dict(optics=opticsDf, particles=particlesDf.iloc[subset1Idxs, :]), starOutFnameTemplate%1, overwrite=True)
        starfile.write(dict(optics=opticsDf, particles=particlesDf.iloc[subest2Idxs, :]), starOutFnameTemplate%2, overwrite=True)

    return subset1Idxs, subest2Idxs

def split_particles_based_on_angles(particlesDf, n_chunks, n_cpus=1, so3ClassIdName='so3ClassId', hp_order=2):

    if n_chunks == 1:
        return [particlesDf.copy()]

    assert n_chunks < 4_000, "Error, this won't work with that many classes"
    print(f"Splitting particles into {n_chunks} chunks")

    n_particles = particlesDf.shape[0]
    repeated_indices, n_copies = _get_repeated_image_idxs(particlesDf)

    repeated_indices = np.array(repeated_indices, dtype=np.int64)
    particlesDfList = [particlesDf.iloc[repeated_indices[:,i], :].copy() for i in range(repeated_indices.shape[-1])]

    particlesDf = particlesDfList[0]

    discretizer = SO3_discretizer(hp_order=hp_order)
    angles = particlesDf[RELION_ANGLES_NAMES].values
    # particlesDf = particlesDf.copy()
    particlesDf[so3ClassIdName] = discretizer.eulerDegs_to_idx(angles)
    group_sizes = particlesDf.groupby(so3ClassIdName).size().reset_index(name='size')

    traversal_order = np.fromiter(discretizer.traverse_so3_sphere(group_sizes[so3ClassIdName].values, n_jobs=n_cpus), dtype=np.int64)
    # Sorting groups by size in descending order to distribute them evenly
    group_sizes = group_sizes.set_index(so3ClassIdName).loc[traversal_order].reset_index()

    # Calculate the approximate size of each chunk
    total_size = particlesDf.shape[0]
    approx_chunk_size = total_size // n_chunks

    chunks = []  # To store the resulting chunks
    current_chunk = []
    current_size = 0

    for so3ClassId in traversal_order:
        class_size = group_sizes.loc[group_sizes[so3ClassIdName] == so3ClassId, 'size'].values[0]
        if current_size + class_size > approx_chunk_size and current_chunk:
            # Add the current chunk to chunks and start a new one
            chunks.append(pd.concat(current_chunk, ignore_index=True))
            current_chunk = []
            current_size = 0
        # Add the current class's rows to the current chunk
        selection_mask = (particlesDf[so3ClassIdName] == so3ClassId).values
        _c_chunk = pd.concat([x[selection_mask] for x in particlesDfList])
        _c_chunk.drop(so3ClassIdName, axis=1)
        current_chunk.append(_c_chunk)
        current_size += class_size

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(pd.concat(current_chunk, ignore_index=True))

    # Adjust the last chunks if needed to ensure all data is included
    while len(chunks) > n_chunks:
        # Move the last chunk's data to the second last chunk
        chunks[-2] = pd.concat([chunks[-2], chunks[-1]], ignore_index=True)
        del chunks[-1]
    assert sum([len(x) for x in chunks]) == n_particles
    return chunks

def _get_repeated_image_idxs(df):

    #Deal with duplicate particles, like symmetry expanded ones
    duplicates = df.reset_index(drop=True).groupby(RELION_IMAGE_FNAME).groups
    repeated_indices = [list(indices) for name, indices in duplicates.items()]
    n_copies = set([len(x) for x in repeated_indices])
    assert len(n_copies) == 1, f"Error, different number of duplicated inputs {n_copies}"
    n_duplication = n_copies.pop()
    return repeated_indices, n_duplication

def sort_particles_based_on_angles(particles:PARTICLES_SET_OR_STARFNAME, n_cpus=1, so3ClassIdName='so3ClassId', hp_order=3):

    so3ClassIdName += "_%d"%hp_order

    if isinstance(particles, ParticlesStarSet):
        particlesDf = particles.particles_md
    else:
        particlesDf = particles


    repeated_indices, n_copies = _get_repeated_image_idxs(particlesDf)

    repeated_indices = np.array(repeated_indices, dtype=np.int64)
    particlesDfList = [particlesDf.iloc[repeated_indices[:,i], :] for i in range(repeated_indices.shape[-1])]

    particlesDf = particlesDfList[0]
    angles = particlesDf[RELION_ANGLES_NAMES].values

    discretizer = SO3_discretizer(hp_order=hp_order)
    particlesDf[so3ClassIdName] = discretizer.eulerDegs_to_idx(angles)
    full_nodes_traversal_order = np.fromiter(discretizer.traverse_so3_sphere(n_jobs=n_cpus), dtype=np.int64)
    traversal_order = traverse_node_subset(full_nodes_traversal_order, particlesDf[so3ClassIdName].values) #TODO: Make this more efficient by dropping duplicates
    traversal_order_df = pd.Categorical(particlesDf[so3ClassIdName],
                                        categories=traversal_order[np.sort(np.unique(traversal_order,
                                                            return_index=True)[1])], ordered=True).argsort()

    if isinstance(particles, ParticlesStarSet):
        particles = particles.createSubset(idxs=traversal_order_df)
    else:
        particles = particlesDf.iloc[traversal_order_df, :]

    return particles


def traverse_node_subset(full_path, subset_nodes):

    sorted_unique_fullNodes, unique_indices_fullNodes = np.unique(full_path, return_index=True)
    return _traverse_subset_loop(sorted_unique_fullNodes, unique_indices_fullNodes, subset_nodes)

@numba.jit(nopython=True)
def _traverse_subset_loop(sorted_unique_fullNodes, unique_indices_fullNodes, subset_nodes):

    # subset_nodes = np.sort(subset_nodes)
    subset_nodes_sortIdxs = np.argsort(subset_nodes)
    subset_nodes = subset_nodes[subset_nodes_sortIdxs]

    results_idxs = np.full_like(subset_nodes, -1)
    n_results = len(results_idxs)
    j = 0
    finished = False
    for i, full in enumerate(sorted_unique_fullNodes):
        sub = subset_nodes[j]
        while sub <= full:
            results_idxs[j] = unique_indices_fullNodes[i]
            j += 1
            if j >= n_results:
                finished = True
                break
            sub = subset_nodes[j]
        if finished:
            break
    sort_idxs = np.argsort(results_idxs)
    results = subset_nodes[sort_idxs]
    return results

def _test_traverse():
    # Example usage:
    full_path = np.array([1, 2, 3, 7, 8, 4, 5, 6, 6]) #np.array([1, 2, 3, 4, 5, 2, 6, 7, 3, 8])
    subset_nodes = np.array([3, 2, 5, 5, 6, 1, 1, 6, 8])

    result = traverse_node_subset(full_path, subset_nodes)
    print(result)  # Output: [2 3 2 3 8]

def _test_split():
    import starfile
    parts = starfile.read("/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star")["particles"]
    split_particles_based_on_angles(parts, 4)
    print()

def _test_sort():
    import starfile
    parts = starfile.read("/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/allparticles.star")["particles"]
    out = sort_particles_based_on_angles(parts, n_cpus=1, so3ClassIdName='so3ClassId', hp_order=2)
    print(out)
    print()
if __name__ == "__main__":
    _test_sort()