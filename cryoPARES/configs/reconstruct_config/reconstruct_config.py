from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Reconstruct_config:

    disable_compile_insert_central_slices_rfft_3d_multichannel: bool = False
    compile_insert_central_slices_rfft_3d_multichanne_mode: Optional[str] = None #"max-autotune" does not work

    eps: float = 1e-3 #epsilon in the denominator. ~ tikhonov regularization
    weight_with_confidence: bool = False
    correct_ctf: bool = True
