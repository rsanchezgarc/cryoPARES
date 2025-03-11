from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class NnetInference_config:
    n_dataworkers: int = 0
    batch_size: int = 32