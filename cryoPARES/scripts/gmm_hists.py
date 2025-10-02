import os
from typing import Optional, List

import starfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from scipy import stats, optimize
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

from cryoPARES.constants import DIRECTIONAL_ZSCORE_NAME, RELION_EULER_CONVENTION, RELION_ANGLES_NAMES
from cryoPARES.geometry.metrics_angles import rotation_error_with_sym


def compare_prob_hists(fname_good: List[str], fname_all: Optional[List[str]],
                       fname_ignore: Optional[List[str]] = None,
                       score_name: str = DIRECTIONAL_ZSCORE_NAME, out_img_fname: Optional[str] = None,
                       symmetry: Optional[str] = None, degs_error_thr: Optional[float] = 3., bins: int = 200,
                       fraction_to_sample_from_all: Optional[float] = None,
                       compute_gmm: bool = True
                       ):
    """
        @param fname_good: the name of the star file with good particles
        @param fname_all: the name of the star file with all particles. It is a superset of fname_good
        @param fname_ignore: the name of the star file with a subset of fname_all particles to be ignored
        @param score_name: The name of the score to plot
        @param out_img_fname:
        @param symmetry: Only required if fname_all is not provided
        @param degs_error_thr: Only used if fname_all is not provided. Good particles will be those with angular error smaller than the desired threshold
        @param bins: The number of bins for the histogram
        @param fraction_to_sample_from_all:
        @param compute_gmm: If True, computes the gmm threshold and shows the plot
    """

    # Verify all files exist
    for f in fname_good:
        assert os.path.isfile(f), f"Error, {f} is not a file"

    if fname_ignore:
        for f in fname_ignore:
            assert os.path.isfile(f), f"Error, {f} is not a file"

    def read_multiple(fnames):
        all_parts = []
        for fname in fnames:
            parts = starfile.read(fname)["particles"]
            # print(fname)
            # print(parts[score_name].dtype)
            # breakpoint()
            parts["rlnMicrographName"] = parts["rlnMicrographName"].map(lambda x: os.path.basename(x))
            all_parts.append(parts)
        return pd.concat(all_parts, ignore_index=True)

    parts_good = read_multiple(fname_good)
    print("parts_good", parts_good.shape)

    if fname_all is not None:
        for f in fname_all:
            assert os.path.isfile(f), f"Error, {f} is not a file"
        parts_all = read_multiple(fname_all)
        print("parts_all", parts_all.shape)
    else:
        assert symmetry is not None, "Error, symmetry needs to be provided"
        # We are going to extract bad particles from parts_good

        from scipy.spatial.transform import Rotation as R
        rots = R.from_euler(RELION_EULER_CONVENTION, parts_good[RELION_ANGLES_NAMES],
                            degrees=True)
        gtRots = R.from_euler(RELION_EULER_CONVENTION, parts_good[[x+"_ori" for x in RELION_ANGLES_NAMES]],
                              degrees=True)

        angle = rotation_error_with_sym(torch.as_tensor(rots.as_matrix(), dtype=torch.float32),
                                        torch.as_tensor(gtRots.as_matrix(), dtype=torch.float32),
                                         symmetry=symmetry,)
        angle = torch.rad2deg(angle)
        # plt.hexbin(angle, parts_good[score_name], bins="log") ;plt.show()
        parts_all = parts_good.copy()
        parts_good = parts_good[(angle < degs_error_thr).numpy()]

    # ['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY'] are the primary key columns
    if fname_ignore is not None:
        parts_ignore = read_multiple(fname_ignore)
        print("parts_ignore", parts_ignore.shape)

        # Merge with indicator to identify particles to remove
        merged_ignore = pd.merge(parts_all, parts_ignore,
                                 on=['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY'],
                                 how='left', indicator=True)
        # Keep only particles that are not in the ignore file
        parts_all = merged_ignore[merged_ignore['_merge'] == 'left_only'].drop(columns=['_merge'])
        for col in parts_all.columns:
            if col.endswith('_x'):
                base_col = col[:-2]
                parts_all[base_col] = parts_all[col]
                parts_all.drop(columns=[col], inplace=True)
            elif col.endswith('_y'):
                parts_all.drop(columns=[col], inplace=True)
        print("parts_all after ignore", parts_all.shape)

    if fraction_to_sample_from_all is not None:
        parts_all = parts_all.sample(frac=fraction_to_sample_from_all)
        print("parts_all after sample", parts_all.shape)

    # Merge the dataframes with an indicator
    merged_df = pd.merge(parts_good, parts_all,
                         on=['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY'],
                         how='outer', indicator=True)
    rows_in_df2_not_in_df1 = merged_df[merged_df['_merge'] == 'right_only']
    # Dropping the merge indicator column to get the original dataframe structure
    rows_in_df2_not_in_df1 = rows_in_df2_not_in_df1.drop(columns=['_merge'])

    print(rows_in_df2_not_in_df1.shape)
    rows_in_df2_not_in_df1[score_name] = rows_in_df2_not_in_df1[score_name + '_y']
    assert len(parts_all) > len(rows_in_df2_not_in_df1), "Error, no particles were filtered as bad!!"
    print("parts_good", parts_good[score_name].describe())
    print("parts_all", parts_all[score_name].describe())
    print("parts_bad", rows_in_df2_not_in_df1[score_name].describe())
    if compute_gmm:
        print("gmm threshold", separate_gmm_threshold(rows_in_df2_not_in_df1[score_name].values,
                                                      parts_good[score_name].values)[0])
    # print("parts_good percentiles", np.percentile(parts_good[score_name], [1,5,10]))
    print("parts_all quantile for parts_good", np.quantile(parts_all[score_name], 1 - len(parts_good) / len(parts_all)))

    parts_good.replace([np.inf, -np.inf], np.nan, inplace=True)
    parts_good.dropna(inplace=True, subset=score_name)
    rows_in_df2_not_in_df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_in_df2_not_in_df1.dropna(inplace=True, subset=score_name)

    # assert len(rows_in_df2_not_in_df1) > len(parts_good), ("Error, the set of selected good particles must be larger that"
    #                                                    f" the set of bad particles {(len(rows_in_df2_not_in_df1) , len(parts_good))}")
    # rows_in_df2_not_in_df1[score_name] = np.clip(rows_in_df2_not_in_df1[score_name], -6, 6)
    # parts_good[score_name] = np.clip(parts_good[score_name], -6, 6)

    fig, ax1 = plt.subplots()
    # First dataframe on primary y-axis

    score1 = rows_in_df2_not_in_df1[score_name]
    score2 = parts_good[score_name]

    histout1 = ax1.hist(_clip_percentiles(score1), bins=bins, alpha=0.5, color='orange', label="bad")
    ax1.set_xlabel('Score')
    ax1.set_ylabel('# bad particles')
    ax1.legend(loc='upper left')

    # Second dataframe on secondary y-axis
    ax2 = ax1.twinx()

    histout2 = ax2.hist(_clip_percentiles(score2), bins=bins, alpha=0.5, color='blue', label="good")
    ax2.set_ylabel('# good particles')
    ax2.legend(loc='upper right')

    # fraction_value = 1e-2
    # true1 = np.where(histout1[0] > rows_in_df2_not_in_df1.shape[0]*fraction_value)[0]
    # true2 = np.where(histout2[0] > parts_good.shape[0]*fraction_value)[0]
    # lowerBound = min(histout1[1][min(0, true1.min()-1)],              histout2[1][min(0, true2.min()-1)])
    # upperBound = max(histout1[1][(true1.max() + 1)%len(histout1[1])], histout2[1][(true2.max() + 1)%len(histout2[1])])

    ##########################################
    # lowerBound = np.percentile(rows_in_df2_not_in_df1[score_name], 0.1)
    # upperBound = np.percentile(parts_good[score_name], 99.9)

    lowerBound = _clip_percentiles(score1).min()
    upperBound = _clip_percentiles(score2).max()

    plt.xlim(lowerBound, upperBound)
    if out_img_fname is not None:
        plt.savefig(out_img_fname)
    plt.show()


def find_intersection(mean1, std1, weight1, mean2, std2, weight2):
    """
    Find the intersection point between two Gaussian curves
    """

    def gaussian_diff(x):
        return weight1 * stats.norm.pdf(x, mean1, std1) - weight2 * stats.norm.pdf(x, mean2, std2)

    # Use mean of means as initial guess
    x0 = (mean1 + mean2) / 2

    try:
        result = optimize.root_scalar(gaussian_diff, x0=x0,
                                      bracket=[min(mean1, mean2), max(mean1, mean2)])
        if result.converged:
            return result.root
    except:
        pass

    return None


def separate_gmm_threshold(dist1, dist2):
    """
    Fit separate GMMs to each distribution and find intersection between relevant components
    """
    # Reshape data for GMM
    X1 = dist1.reshape(-1, 1)  # bad
    X2 = dist2.reshape(-1, 1)  # good
    X1 = np.nan_to_num(X1, posinf=np.nanmax(X1[~np.isinf(X1)]), neginf=np.nanmin(~np.isinf(X1)))
    X2 = np.nan_to_num(X2, posinf=np.nanmax(~np.isinf(X2)), neginf=np.nanmin(~np.isinf(X2)))

    x1_for_plot = _clip_percentiles(X1.flatten())
    x2_for_plot = _clip_percentiles(X2.flatten())

    X1 = np.clip(X1, a_min=None, a_max=np.percentile(X2, 99))
    X2 = np.clip(X2, a_min=np.percentile(X1, 1), a_max=None)

    # Fit GMMs with 2 components to each distribution
    gmm1 = GaussianMixture(n_components=2, random_state=42)
    gmm2 = GaussianMixture(n_components=2, random_state=42)

    gmm1.fit(X1)
    gmm2.fit(X2)

    # Sort components by mean for each GMM
    means1 = gmm1.means_.flatten()
    means2 = gmm2.means_.flatten()

    weights1 = gmm1.weights_.flatten()
    weights2 = gmm2.weights_.flatten()

    stds1 = np.sqrt(gmm1.covariances_.flatten())
    stds2 = np.sqrt(gmm2.covariances_.flatten())

    # Sort components
    sort_idx1 = np.argsort(means1)
    sort_idx2 = np.argsort(means2)

    means1 = means1[sort_idx1]
    means2 = means2[sort_idx2]
    weights1 = weights1[sort_idx1]
    weights2 = weights2[sort_idx2]
    stds1 = stds1[sort_idx1]
    stds2 = stds2[sort_idx2]

    # Determine which distribution is "larger" based on their dominant components
    larger_dist_mean = max(means1[np.argmax(weights1)], means2[np.argmax(weights2)])

    if larger_dist_mean == means1[np.argmax(weights1)]:
        smaller_means, larger_means = means2, means1
        smaller_weights, larger_weights = weights2, weights1
        smaller_stds, larger_stds = stds2, stds1
        smaller_gmm, larger_gmm = gmm2, gmm1
    else:
        smaller_means, larger_means = means1, means2
        smaller_weights, larger_weights = weights1, weights2
        smaller_stds, larger_stds = stds1, stds2
        smaller_gmm, larger_gmm = gmm1, gmm2

    # Try to find intersection between relevant components
    intersection = find_intersection(
        smaller_means[0], smaller_stds[0], smaller_weights[0],
        larger_means[1], larger_stds[1], larger_weights[1]
    )

    # If intersection found, use it; otherwise, use midpoint
    if intersection is not None:
        threshold = intersection
        threshold_method = "intersection"
    else:
        threshold = (smaller_means[0] + larger_means[1]) / 2
        threshold_method = "midpoint"

    plot_separate_gmm_distributions(x1_for_plot, x2_for_plot, threshold, (smaller_gmm, larger_gmm), threshold_method)
    return threshold, (smaller_gmm, larger_gmm), threshold_method


def _clip_percentiles(score):
    low = 0.1  # 1 #0.1
    up = 100 - low
    return np.clip(score, np.percentile(score, low), np.percentile(score, up))


def plot_separate_gmm_distributions(dist1, dist2, threshold, gmms, threshold_method):
    plt.figure(figsize=(15, 6))

    # Plot histograms
    plt.hist(dist1, bins=50, alpha=0.5, density=True, label='Distribution 1')
    plt.hist(dist2, bins=50, alpha=0.5, density=True, label='Distribution 2')

    # Plot individual GMM components
    x_range = np.linspace(min(min(dist1), min(dist2)), max(max(dist1), max(dist2)), 1000)

    colors = ['--r', '--b']
    for i, gmm in enumerate(gmms):
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_.flatten()

        sort_idx = np.argsort(means)
        means = means[sort_idx]
        stds = stds[sort_idx]
        weights = weights[sort_idx]

        for j, (mean, std, weight) in enumerate(zip(means, stds, weights)):
            pdf = weight * stats.norm.pdf(x_range, mean, std)
            plt.plot(x_range, pdf, colors[i], alpha=0.5,
                     label=f'Dist {i + 1} Component {j + 1} (μ={mean:.2f}, σ={std:.2f}, w={weight:.2f})')

    # Plot threshold
    plt.axvline(x=threshold, color='g', linestyle='-',
                label=f'Threshold ({threshold:.2f}, {threshold_method})')

    plt.title('GMM Fitting')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    # plt.xlim(-6, 6)
    plt.show()


def _test_gmm():
    def generate_mixed_sample_data(n_samples=1000):
        # Generate two distributions, each a mixture of "good" and "bad" points
        dist1_good = np.random.normal(loc=3, scale=0.8, size=int(n_samples * 0.7))
        dist1_bad = np.random.normal(loc=5, scale=1.2, size=int(n_samples * 0.3))
        dist1 = np.concatenate([dist1_good, dist1_bad])

        dist2_good = np.random.normal(loc=7, scale=1.0, size=int(n_samples * 0.8))
        dist2_bad = np.random.normal(loc=5, scale=1.5, size=int(n_samples * 0.2))
        dist2 = np.concatenate([dist2_good, dist2_bad])

        return dist1, dist2

    # Generate sample data
    dist1, dist2 = generate_mixed_sample_data()

    # Calculate threshold using separate GMMs
    threshold, gmms, threshold_method = separate_gmm_threshold(dist1, dist2)

    # Plot distributions and threshold
    plot_separate_gmm_distributions(dist1, dist2, threshold, gmms, threshold_method)

    # Evaluate threshold
    true_labels = np.concatenate([np.zeros_like(dist1), np.ones_like(dist2)])
    predicted_labels = np.concatenate([dist1 > threshold, dist2 > threshold])
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Threshold: {threshold:.4f} (using {threshold_method})")
    print(f"Accuracy: {accuracy:.4f}")


def _test():
    outdir = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/tmp/zscorePlots"

    ## BGAL
    # fname_good = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/BGAL/inferecence/trained_on_apo04882/lig_00892/top1_local4_1_filter3/nnet_predictions_zscore.star"
    # fname_all = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/BGAL/inferecence/trained_on_apo04882/all_lig_00892/top1_local4_1_filter3/nnet_predictions_zscore.star"
    # outname = os.path.join(outdir, "bgal_lig00892_trained_on_apo04882.png")

    # fname_good = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/BGAL/inferecence/trained_on_apo04882/lig_00893/top1_local4_1_filter3/nnet_predictions_zscore.star"
    # fname_all = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/BGAL/inferecence/trained_on_apo04882/all_lig_00893/top1_local4_1_filter3/nnet_predictions_zscore.star"
    # outname = os.path.join(outdir, "bgal_lig00893_trained_on_apo04882.png")

    ## PKM2
    # fname_good = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/PKM2/inferecence/trained_on_astex5534/apo_01061/top1_local6_2_filter3/nnet_predictions.star"
    # fname_all = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/PKM2/inferecence/trained_on_astex5534/all_apo_01061/nnet_predictions.star"
    # outname = os.path.join(outdir, "PKM2_apo01061_trained_on_astex5534.png")

    fname_good = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/PKM2/inferecence/trained_on_astex5534/lig_01029/top1_local6_1_filter3/nnet_predictions.star"
    fname_all = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/PKM2/inferecence/trained_on_astex5534/all_lig_01029/only_inference/nnet_predictions.star"
    outname = os.path.join(outdir, "PKM2_lig01029_trained_on_astex5534.png")

    ## TRPML
    # fname_all = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/TRPML1/inferecence/new_3d_class_on_lig_04768/all_lig_05811/nnet_predictions_zscore.star"
    # fname_good = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/TRPML1/inferecence/new_3d_class_on_lig_04768/lig_05811/nnet_predictions_zscore.star"
    # outname = os.path.join(outdir, "TRPML_lig05811_trained_on_lig04768.png")

    # fname_all = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/TRPML1/inferecence/new_3d_class_on_lig_04768/all_apo_06195/nnet_predictions_zscore.star"
    # fname_good = "/tmp_mnt/filer1/cryo-em/manual_processing/rsgarcia/cryo/data/supervisedAngles/RESULTS_BENCHMARK/TRPML1/inferecence/new_3d_class_on_lig_04768/apo_06195/nnet_predictions_zscore.star"
    # outname = os.path.join(outdir, "TRPML_apo0619_trained_on_lig04768.png")

    compare_prob_hists([fname_good], [fname_all], out_img_fname=outname)


if __name__ == "__main__":
    # _test()
    from argParseFromDoc import parse_function_and_call

    parse_function_and_call(compare_prob_hists)
