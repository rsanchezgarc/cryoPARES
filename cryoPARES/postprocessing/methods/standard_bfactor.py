"""
Standard B-factor post-processing (Guinier sharpening + FSC weighting).
"""
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryoPARES.postprocessing.methods.base import PostProcessor
from cryoPARES.postprocessing.guinier import estimate_bfactor
from cryoPARES.postprocessing.sharpening import apply_bfactor_and_fsc_weight


class BfactorPostProcessor(PostProcessor):
    """B-factor sharpening + FSC-weighting post-processor."""

    def process(self, half1: np.ndarray, half2: np.ndarray,
                avg_map: np.ndarray, mask: np.ndarray,
                fsc_curves: dict, px_A: float,
                output_dir: str,
                adhoc_bfac: Optional[float] = None,
                guinier_lo_A: float = 10.0,
                lowpass_A: Optional[float] = None,
                save_guinier_plot: Optional[str] = None) -> np.ndarray:
        """
        Apply Guinier B-factor estimation and FSC-weighted sharpening.
        """
        # B-factor estimation + sharpening cutoff
        bfactor, slope, intercept, x_guin, y_guin, valid_mask, cutoff_A = \
            estimate_bfactor(fsc_curves, avg_map, px_A, guinier_lo_A, adhoc_bfac)
        print(f"B-factor: {bfactor:.1f} Å²")

        # Determine lowpass: user override → else first-zero-crossing from Guinier
        # (passed as lowpass_A=None lets apply_bfactor_and_fsc_weight compute its own
        # first-zero-crossing; passing the explicit cutoff_A is equivalent but allows
        # the caller to override it).
        lp = lowpass_A
        if lp is None and np.isfinite(cutoff_A):
            lp = float(cutoff_A)
            print(f"Applying hard Fourier cutoff at first-zero-crossing: {lp:.2f} Å")
        elif lp is not None:
            print(f"Applying hard Fourier cutoff at user-specified: {lp:.2f} Å")

        # Sharpening
        sharpened = apply_bfactor_and_fsc_weight(
            avg_map,
            fsc_curves["fsc_corrected"],
            fsc_curves["spatial_freq"],
            bfactor, px_A,
            lowpass_A=lp)

        # Save Guinier data
        if adhoc_bfac is None and len(x_guin) > 0:
            guin_txt = os.path.join(output_dir, "guinier_data.txt")
            fitted_y = slope * x_guin + intercept
            data = np.column_stack([x_guin, y_guin, fitted_y])
            np.savetxt(guin_txt, data, delimiter=",",
                       header="inv_res_sq,ln_amp,fitted_line",
                       comments="")
            print(f"Wrote: {guin_txt}")

            # Guinier plot
            if save_guinier_plot is None:
                save_guinier_plot = os.path.join(output_dir, "guinier_plot.png")
            self._save_guinier_plot(
                x_guin, y_guin, valid_mask, slope, intercept, bfactor,
                save_guinier_plot)
            print(f"Wrote: {save_guinier_plot}")

        return sharpened

    @staticmethod
    def _save_guinier_plot(x: np.ndarray, y: np.ndarray,
                           valid_mask: np.ndarray,
                           slope: float, intercept: float,
                           bfactor: float, path: str):
        """Save Guinier plot: scatter + linear fit."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(x[valid_mask], y[valid_mask], s=10, color="steelblue",
                   alpha=0.7, label="data (fit region)")
        ax.scatter(x[~valid_mask], y[~valid_mask], s=5, color="lightgray",
                   alpha=0.5, label="data (excluded)")

        x_fit = x[valid_mask]
        if x_fit.size > 0:
            x_line = np.array([x_fit.min(), x_fit.max()])
            ax.plot(x_line, slope * x_line + intercept, "r-",
                    linewidth=2, label=f"fit (B={bfactor:.1f} Å²)")

        ax.set_xlabel("s² (1/Å²)")
        ax.set_ylabel("ln(amplitude)")
        ax.set_title(f"Guinier plot  |  B = {bfactor:.1f} Å²")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def postprocess_bfactor(
    half1: str,
    half2: str,
    output_dir: str,
    mask: Optional[str] = None,
    auto_mask: bool = False,
    auto_mask_threshold: Optional[float] = None,
    auto_mask_dilation_A: float = 8.0,
    auto_mask_edge_width: int = 6,
    auto_mask_spherical: bool = False,
    px_A: Optional[float] = None,
    adhoc_bfac: Optional[float] = None,
    guinier_lo_A: float = 10.0,
    lowpass_A: Optional[float] = None,
    save_fsc_plot: Optional[str] = None,
    save_guinier_plot: Optional[str] = None,
):
    """
    Standard B-factor post-processing of two cryo-EM half-maps.

    :param half1: Path to half-map 1 (.mrc)
    :param half2: Path to half-map 2 (.mrc)
    :param output_dir: Directory for all output files
    :param mask: Path to mask .mrc (provide --mask or use --auto_mask)
    :param auto_mask: Auto-generate a soft mask from the average half-map
    :param auto_mask_threshold: Density threshold for auto-mask binarisation (default: Otsu)
    :param auto_mask_dilation_A: Dilation radius in Å added around thresholded region (default 8.0)
    :param auto_mask_edge_width: Soft-edge width in pixels (default 6)
    :param auto_mask_spherical: Use a spherical soft mask instead of a density-threshold mask
    :param px_A: Pixel size in Å — overrides the MRC header value if given
    :param adhoc_bfac: Manual B-factor in Å²; skips automatic Guinier plot estimation
    :param guinier_lo_A: Lower resolution limit in Å for the Guinier linear fit (default 10.0)
    :param lowpass_A: Apply a final low-pass filter at this resolution in Å (default: FSC=0.143 resolution)
    :param save_fsc_plot: Path to save FSC curve plot; default output_dir/fsc_plot.png
    :param save_guinier_plot: Path to save Guinier plot; default output_dir/guinier_plot.png
    """
    BfactorPostProcessor().run_pipeline(
        half1_path=half1,
        half2_path=half2,
        output_dir=output_dir,
        mask_path=mask,
        auto_mask=auto_mask,
        auto_mask_threshold=auto_mask_threshold,
        auto_mask_dilation_A=auto_mask_dilation_A,
        auto_mask_edge_width=auto_mask_edge_width,
        auto_mask_spherical=auto_mask_spherical,
        px_A=px_A,
        save_fsc_plot=save_fsc_plot,
        adhoc_bfac=adhoc_bfac,
        guinier_lo_A=guinier_lo_A,
        lowpass_A=lowpass_A,
        save_guinier_plot=save_guinier_plot,
    )
