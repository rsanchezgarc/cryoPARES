from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Projmatching_config:

    grid_distance_degs: float = 12.0
    grid_step_degs: float = 2.0
    filter_resolution_angst: float = 6.
    max_shift_fraction: float = 0.2
    correct_ctf: bool = True

    disable_compile_projectVol: bool = True
    compile_projectVol_mode: Optional[str] = None #"max-autotune"

    disable_compile_correlate_dft_2d: bool = True
    compile_correlate_dft_2d_mode: Optional[str] = "max-autotune"

    disable_compile_analyze_cc: bool = True #You need to use torch.compiler.cudagraph_mark_step_begin() before the execution of the alinging step
    compile_analyze_cc_mode: Optional[str] = None #"max-autotune" #None #"reduce-overhead" #"max-autotune"
