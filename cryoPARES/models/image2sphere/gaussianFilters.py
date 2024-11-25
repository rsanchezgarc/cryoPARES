import os.path
from math import ceil
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import numpy as np
from typing import Optional, List

from einops import einops
from omegaconf import MISSING

from cryoPARES.configManager.config_searcher import inject_config


@inject_config(classname="GaussianFilterBank")
class NaiveGaussianFilterBank(nn.Module):

    def __init__(self, in_channels, sigma_values, kernel_sizes: Optional[List[int]], out_dim: Optional[int] = None):
        super(NaiveGaussianFilterBank, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [ceil(4*sigma +1) for sigma in sigma_values]
        max_kernel_size = max(kernel_sizes)

        self.in_channels = in_channels
        self.sigma_values = sigma_values
        self.kernel_sizes = kernel_sizes
        # Create a list to hold the Gaussian kernels
        kernels = []
        for kernel_size, sigma in zip(kernel_sizes, sigma_values):
            # Create 1D Gaussian kernel

            if sigma > 0:
                kernel_1D = self.gaussian_kernel(kernel_size, sigma)
                # Create 2D Gaussian kernel
                kernel_2D = kernel_1D[:, None] * kernel_1D[None, :]
            else:
                kernel_2D = torch.tensor([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0]])
                kernel_size = 3

            # Pad kernel with zeros to have the same size as max_kernel_size
            padding = max_kernel_size - kernel_size
            pad_left = padding // 2
            pad_right = padding - pad_left
            kernel_2D = F.pad(kernel_2D, (pad_left, pad_right, pad_left, pad_right))

            # Add channel dimension and append to the list
            kernels.append(kernel_2D[None, None, :, :])

        # Stack the Gaussian kernels into a single tensor
        all_kernels = torch.cat(kernels, dim=0)

        # Repeat the kernels for each channel in the image
        all_kernels = all_kernels.repeat(self.in_channels, 1, 1, 1)

        # Register as a buffer so PyTorch can recognize it as a model parameter
        self.register_buffer('all_kernels', all_kernels)

        if out_dim and out_dim != self.all_kernels.shape[0]:
            self.lastLayer = nn.Conv2d(in_channels=self.all_kernels.shape[0], out_channels=out_dim,
                                       kernel_size=max(kernel_sizes), padding="same")
        else:
            self.lastLayer = nn.Identity()
    @staticmethod
    def gaussian_kernel(size, sigma):
        coords = torch.arange(size).float()
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def forward(self, x):
        # Perform depthwise convolution
        x = F.conv2d(x, self.all_kernels, groups=self.in_channels, padding="same")
        return self.lastLayer(x)

@inject_config(classname="GaussianFilterBank")
class SeparableGaussianFilterBank(nn.Module):
    def __init__(self, in_channels, sigma_values, kernel_sizes: Optional[List[int]] = None,
                 out_dim: Optional[int] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [ceil(4 * sigma + 1) for sigma in sigma_values]

        self.in_channels = in_channels
        self.n_filters = len(sigma_values)
        self.sigma_values = sigma_values
        max_kernel_size = max(kernel_sizes)

        # Create base kernels [n_filters, kernel_size]
        kernels = []
        for kernel_size, sigma in zip(kernel_sizes, sigma_values):
            if sigma > 0:
                kernel = self.gaussian_kernel(kernel_size, sigma)
                padding = max_kernel_size - kernel_size
                pad_left = padding // 2
                pad_right = padding - pad_left
                kernel = F.pad(kernel, (pad_left, pad_right))
            else:
                kernel = torch.zeros(max_kernel_size)
                kernel[max_kernel_size // 2] = 1.0
            kernels.append(kernel)
        kernels = torch.stack(kernels)  # [n_filters, kernel_size]

        # kernels_h: [(n_filters * in_channels), 1, kernel_size, 1]
        kernels_h = kernels.unsqueeze(1).unsqueeze(-1).repeat_interleave(in_channels, dim=0)

        # kernels_v: [(n_filters * in_channels), 1, 1, kernel_size]
        kernels_v = kernels.unsqueeze(1).unsqueeze(2).repeat_interleave(in_channels, dim=0)

        self.register_buffer('kernels_h', kernels_h)
        self.register_buffer('kernels_v', kernels_v)
        self.out_projection = nn.Conv2d(self.n_filters * in_channels, out_dim, 1) if out_dim else nn.Identity()

    @staticmethod
    def gaussian_kernel(size, sigma):
        coords = torch.arange(size).float()
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g

    def forward(self, x):
        # Apply horizontal: [b, (f*c), h, w]
        h_filtered = F.conv2d(x, self.kernels_h, groups=self.in_channels, padding='same')

        # Apply vertical: [b, (f*c), h, w]
        v_filtered = F.conv2d(h_filtered, self.kernels_v, groups=self.in_channels * self.n_filters, padding='same')

        return self.out_projection(v_filtered)

@inject_config(classname="GaussianFilterBank")
class FFTGaussianFilterBank(nn.Module):
    def __init__(self, in_channels, sigma_values, kernel_sizes: Optional[List[int]] = None,
                 out_dim: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        assert kernel_sizes is None or kernel_sizes is MISSING, "Error, kernel sizes not used in FFT-based convolution"
        self.sigma_values = sigma_values
        self.out_projection = nn.Conv2d(len(sigma_values) * in_channels, out_dim, 1) if out_dim else nn.Identity()

    def get_gaussian_freq(self, sigma: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Create Gaussian filters in frequency domain for all sigmas at once"""
        rows, cols = shape[-2:]
        center_row, center_col = rows // 2, cols // 2

        y = torch.arange(-center_row, rows - center_row, device=sigma.device)
        x = torch.arange(-center_col, cols - center_col, device=sigma.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        # Compute all filters at once: [n_sigma, H, W]
        sigma = sigma.view(-1, 1, 1)
        freq_resp = torch.where(
            sigma == 0,
            torch.ones((1, rows, cols), device=sigma.device),
            torch.exp(-2 * (np.pi ** 2) * sigma ** 2 * (X ** 2 + Y ** 2) / (rows * cols))
        )
        return freq_resp

    def forward(self, x):
        device = x.device
        sigma_tensor = torch.tensor(self.sigma_values, device=device)

        # Get frequency response for all filters: [n_sigma, H, W]
        H = self.get_gaussian_freq(sigma_tensor, x.shape)

        # FFT of input
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

        # Apply all filters at once using broadcasting
        x_freq = x_freq.unsqueeze(1)  # [B, 1, C, H, W]
        H = H.unsqueeze(1).unsqueeze(0)  # [1, n_sigma, 1, H, W]

        filtered_freq = x_freq * H
        filtered_freq = torch.fft.ifftshift(filtered_freq, dim=(-2, -1))
        filtered = torch.fft.ifft2(filtered_freq).real

        # Rearrange to expected output format
        filtered = einops.rearrange(filtered, 'b f c h w -> b (f c) h w')
        return self.out_projection(filtered)

GaussianFilterBank = SeparableGaussianFilterBank


def _getImg():
    import skimage
    image = skimage.data.camera()

    # Convert to PyTorch tensor and add batch and channel dimensions
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image_tensor

def _getParticle():
    from starstack import ParticlesStarSet
    particles = ParticlesStarSet(os.path.expanduser("~/cryo/data/preAlignedParticles/EMPIAR-10166/data/1000particles.star"))
    img, md = particles[1]
    image_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image_tensor

def _visual_test():
    import matplotlib.pyplot as plt

    # Convert to PyTorch tensor and add batch and channel dimensions
    image_tensor = _getParticle() #_getImg() #

    # Initialize GaussianFilterBank
    sigma_values = [0., 1., 3., 5., 10.]
    # from math import ceil
    # kernel_sizes = [ceil(4*sigma +1) for sigma in sigma_values]
    GaussianFilters = NaiveGaussianFilterBank #SeparableGaussianFilterBank #FFTGaussianFilterBank #SeparableGaussianFilterBank #NaiveGaussianFilterBank
    gaussian_filter_bank = GaussianFilters(1) #GaussianFilters(1, sigma_values, kernel_sizes)

    # Apply the Gaussian filters
    filtered_images = gaussian_filter_bank(image_tensor)

    # Plotting the original and filtered images
    fig, axs = plt.subplots(1, len(gaussian_filter_bank.sigma_values), figsize=(15, 15))

    # Plot filtered images
    for i in range(len(gaussian_filter_bank.sigma_values)):
        axs[i].imshow(filtered_images[0, i].detach().numpy(), cmap='gray')
        axs[i].set_title(f"Filtered with sigma {gaussian_filter_bank.sigma_values[i]}") #{kernel_sizes[i]}x{kernel_sizes[i]} kernel")
        axs[i].axis('off')

    plt.show()


def benchmark_filters(
        original_filter,
        separable_filter,
        batch_sizes: List[int],
        image_sizes: List[int],
        n_repeats: int = 5
) -> Tuple[dict, dict]:

    import torch
    import time
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_original = {}
    results_separable = {}

    for batch_size in batch_sizes:
        for img_size in image_sizes:
            key = f"b{batch_size}_s{img_size}"
            x = torch.randn(batch_size, 1, img_size, img_size).to(device)
            original_filter.to(device)
            separable_filter.to(device)

            # Warmup
            with torch.no_grad():
                original_filter(x)
                separable_filter(x)

            # Time original
            torch.cuda.synchronize()
            times_orig = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                with torch.no_grad():
                    original_filter(x)
                torch.cuda.synchronize()
                times_orig.append(time.perf_counter() - start)

            # Time separable
            torch.cuda.synchronize()
            times_sep = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                with torch.no_grad():
                    separable_filter(x)
                torch.cuda.synchronize()
                times_sep.append(time.perf_counter() - start)

            results_original[key] = np.mean(times_orig)
            results_separable[key] = np.mean(times_sep)

            print(f"Batch {batch_size}, Size {img_size}:")
            print(f"Original: {results_original[key]:.4f}s")
            print(f"Separable: {results_separable[key]:.4f}s")
            print(f"Speedup: {results_original[key] / results_separable[key]:.2f}x\n")

    return results_original, results_separable


def run_benchmark():
    sigma_values = [15.] #[0., 1., 3., 5., 10., 15.]

    original = NaiveGaussianFilterBank(1, sigma_values)
    other = SeparableGaussianFilterBank(1, sigma_values)
    # other = FFTGaussianFilterBank(1, sigma_values)

    batch_sizes = [1, 4, 8, 32]
    image_sizes = [128, 256]

    return benchmark_filters(original, other, batch_sizes, image_sizes)



if __name__ == "__main__":
    _visual_test()
    # run_benchmark()

