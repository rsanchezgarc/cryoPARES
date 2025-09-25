import torch

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.symmetry = "C1"

    def forward(self, x, top_k: int = 1):
        batch_size = x.shape[0]
        # Dummy outputs for pred_rotmats and maxprobs
        pred_rotmats = torch.randn(batch_size, top_k, 3, 3)
        maxprobs = torch.rand(batch_size, top_k)
        
        # The model is expected to return 5 values. The first 3 are ignored by the caller.
        return None, None, None, pred_rotmats, maxprobs
