from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Projmatching_config:

    grid_distance_degs: float = 6.0 #maximum angular distance from the original pose. Grid will go from -grid_distance_degs to grid_distance_degs
    grid_step_degs: float = 2.0
    max_resolution_A: float = 6.
    max_shift_fraction: float = 0.2
    correct_ctf: bool = True
    verbose: bool = False
    top_k_poses_localref: int = 1

    disable_compile_projectVol: bool = False
    compile_projectVol_mode: Optional[str] = "max-autotune"

    disable_compile_correlate_dft_2d: bool = True #at the moment, inductor does not support complex numbers
    compile_correlate_dft_2d_mode: Optional[str] = "max-autotune"

    disable_compile_analyze_cc: bool = False
    compile_analyze_cc_mode: Optional[str] = "max-autotune" #None #"reduce-overhead" #"max-autotune"
