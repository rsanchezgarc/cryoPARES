"""
Abstract base class + shared pipeline for cryo-EM post-processing.
"""
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryoPARES.utils.reconstructionUtils import get_vol, write_vol
from cryoPARES.postprocessing.fsc_utils import run_gold_standard_fsc
from cryoPARES.postprocessing.mask_utils import generate_mask


class PostProcessor(ABC):
    """Abstract base class for cryo-EM post-processing methods."""

    @abstractmethod
    def process(self, half1: np.ndarray, half2: np.ndarray,
                avg_map: np.ndarray, mask: np.ndarray,
                fsc_curves: dict, px_A: float,
                output_dir: str, **method_kwargs) -> np.ndarray:
        """
        Apply the post-processing method.

        Parameters
        ----------
        half1, half2 : (D, H, W) float32 — raw half-maps
        avg_map : (D, H, W) float32 — (half1 + half2) / 2
        mask : (D, H, W) float32 — soft mask in [0, 1]
        fsc_curves : dict — output of run_gold_standard_fsc()
        px_A : float — pixel size in Å
        output_dir : str

        Returns
        -------
        sharpened : np.ndarray (D, H, W) float32
        """

    def run_pipeline(self,
                     half1_path: str,
                     half2_path: str,
                     output_dir: str,
                     mask_path: Optional[str] = None,
                     auto_mask: bool = False,
                     auto_mask_threshold: Optional[float] = None,
                     auto_mask_dilation_A: float = 8.0,
                     auto_mask_edge_width: int = 6,
                     auto_mask_spherical: bool = False,
                     px_A: Optional[float] = None,
                     save_fsc_plot: Optional[str] = None,
                     **method_kwargs):
        """
        Method-independent shared pipeline.

        1. Load half1 / half2 from MRC; validate same shape + voxel size.
        2. Resolve px_A (header or override).
        3. Build/load mask.
        4. Gold-standard FSC.
        5. Print resolution estimates.
        6. avg_map = (half1 + half2) / 2.
        7. self.process() → sharpened map.
        8. Write postprocessed.mrc.
        9. Write fsc_data.txt (6-column CSV).
        10. Save FSC plot.
        11. Write auto_mask.mrc if auto_mask.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1 & 2. Load volumes
        # ------------------------------------------------------------------
        print(f"Loading half-map 1: {half1_path}")
        h1_t, px1 = get_vol(half1_path, pixel_size=None, device="cpu")
        print(f"Loading half-map 2: {half2_path}")
        h2_t, px2 = get_vol(half2_path, pixel_size=None, device="cpu")

        if h1_t.shape != h2_t.shape:
            raise ValueError(
                f"Half-maps have different shapes: {h1_t.shape} vs {h2_t.shape}")

        # Determine pixel size
        if px_A is not None:
            print(f"Using override pixel size: {px_A:.4f} Å/px")
        else:
            if px1 is None or px1 == 0:
                raise ValueError(
                    "Pixel size is 0 in the MRC header. Provide --px_A.")
            if not np.isclose(px1, px2, rtol=1e-3):
                print(f"Warning: half-map pixel sizes differ ({px1} vs {px2}). "
                      f"Using half1 value: {px1}")
            px_A = float(px1)
            print(f"Pixel size from header: {px_A:.4f} Å/px")

        half1 = h1_t.numpy().astype(np.float32)
        half2 = h2_t.numpy().astype(np.float32)

        # ------------------------------------------------------------------
        # 3. Mask
        # ------------------------------------------------------------------
        if mask_path is not None:
            print(f"Loading mask: {mask_path}")
            mask_t, _ = get_vol(mask_path, pixel_size=None, device="cpu")
            mask = mask_t.numpy().astype(np.float32)
        elif auto_mask:
            print("Generating auto-mask …")
            avg_for_mask = (half1 + half2) / 2.0
            mask = generate_mask(
                avg_for_mask, px_A,
                spherical=auto_mask_spherical,
                threshold=auto_mask_threshold,
                dilation_A=auto_mask_dilation_A,
                edge_width_px=auto_mask_edge_width)
            print("Auto-mask generated.")
        else:
            raise ValueError(
                "Provide --mask or use --auto_mask to generate one automatically.")

        # ------------------------------------------------------------------
        # 4. Gold-standard FSC
        # ------------------------------------------------------------------
        print("Computing gold-standard FSC …")
        fsc_curves = run_gold_standard_fsc(half1, half2, mask, px_A)

        # ------------------------------------------------------------------
        # 5. Resolution estimates
        # ------------------------------------------------------------------
        res_0143 = fsc_curves["res_A_0143"]
        res_05   = fsc_curves["res_A_05"]
        res_0143_str = f"{res_0143:.2f} Å" if np.isfinite(res_0143) else "N/A"
        res_05_str   = f"{res_05:.2f} Å"   if np.isfinite(res_05)   else "N/A"
        print(f"Resolution (FSC=0.143): {res_0143_str}")
        print(f"Resolution (FSC=0.5):   {res_05_str}")

        # ------------------------------------------------------------------
        # 6. Average map
        # ------------------------------------------------------------------
        avg_map = (half1 + half2) / 2.0

        # ------------------------------------------------------------------
        # 7. Method-specific processing
        # ------------------------------------------------------------------
        sharpened = self.process(
            half1, half2, avg_map, mask, fsc_curves, px_A,
            output_dir, **method_kwargs)

        # ------------------------------------------------------------------
        # 8. Write postprocessed map
        # ------------------------------------------------------------------
        import torch
        out_mrc = os.path.join(output_dir, "postprocessed.mrc")
        write_vol(torch.from_numpy(sharpened), out_mrc, px_A)
        print(f"Wrote: {out_mrc}")

        # ------------------------------------------------------------------
        # 9. Save fsc_data.txt (6-column CSV)
        # ------------------------------------------------------------------
        fsc_csv = os.path.join(output_dir, "fsc_data.txt")
        data = np.column_stack([
            fsc_curves["spatial_freq"],
            fsc_curves["resolution_A"],
            fsc_curves["fsc_unmasked"],
            fsc_curves["fsc_masked"],
            fsc_curves["fsc_random_masked"],
            fsc_curves["fsc_corrected"],
        ])
        np.savetxt(fsc_csv, data, delimiter=",",
                   header="spatial_freq,res_A,fsc_unmasked,fsc_masked,"
                          "fsc_random_masked,fsc_corrected",
                   comments="")
        print(f"Wrote: {fsc_csv}")

        # ------------------------------------------------------------------
        # 10. FSC plot
        # ------------------------------------------------------------------
        if save_fsc_plot is None:
            save_fsc_plot = os.path.join(output_dir, "fsc_plot.png")
        self._save_fsc_plot(fsc_curves, res_0143, save_fsc_plot)
        print(f"Wrote: {save_fsc_plot}")

        # ------------------------------------------------------------------
        # 11. Auto-mask volume (if generated)
        # ------------------------------------------------------------------
        if auto_mask and mask_path is None:
            mask_out = os.path.join(output_dir, "auto_mask.mrc")
            write_vol(torch.from_numpy(mask), mask_out, px_A)
            print(f"Wrote: {mask_out}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_fsc_plot(fsc_curves: dict, res_0143: float, path: str):
        """Save a 4-curve FSC plot with 0.143 line and resolution annotation."""
        sf  = fsc_curves["spatial_freq"]
        res = fsc_curves["resolution_A"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sf, fsc_curves["fsc_unmasked"],      label="Unmasked",         color="gray",   linestyle="--")
        ax.plot(sf, fsc_curves["fsc_masked"],         label="Masked",           color="steelblue")
        ax.plot(sf, fsc_curves["fsc_random_masked"],  label="Phase-randomised", color="orange", linestyle="-.")
        ax.plot(sf, fsc_curves["fsc_corrected"],      label="Corrected",        color="crimson", linewidth=2)

        ax.axhline(y=0.143, color="red", linestyle=":", linewidth=1, label="FSC=0.143")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(left=0)
        ax.set_xlabel("Spatial frequency (1/Å)")
        ax.set_ylabel("FSC")
        ax.legend(loc="upper right", fontsize=9)

        if np.isfinite(res_0143):
            freq_0143 = 1.0 / res_0143
            ax.axvline(x=freq_0143, color="red", linestyle=":", linewidth=1, alpha=0.7)
            ax.text(freq_0143, 0.2, f" {res_0143:.2f} Å",
                    color="red", fontsize=9, rotation=90, va="bottom")

        # Secondary x-axis in Å
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ticks = ax.get_xticks()
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{1/t:.1f}" if t > 0 else "∞" for t in ticks])
        ax2.set_xlabel("Resolution (Å)")

        ax.set_title("Gold-standard FSC")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
