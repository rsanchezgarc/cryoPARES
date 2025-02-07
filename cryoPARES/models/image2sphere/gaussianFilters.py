import os.path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from typing import Optional, List
from einops import einops
from omegaconf import MISSING

from cryoPARES.configManager.inject_defaults import inject_defaults_from_config, CONFIG_PARAM
from cryoPARES.configs.mainConfig import main_config
from cryoPARES.datamanager.datamanager import get_number_image_channels


class BaseGaussianFilterBank(nn.Module):
    @inject_defaults_from_config(main_config.models.image2sphere.gaussianfilters)
    def __init__(self, in_channels: Optional[int], sigma_values: List[float]= CONFIG_PARAM(),
                 kernel_sizes: Optional[List[int]] = CONFIG_PARAM(), out_channels: Optional[int] = None):
        super().__init__()

        self.in_channels = in_channels if in_channels not in (None, MISSING) else get_number_image_channels()
        self.sigma_values = sigma_values
        self.kernel_sizes = kernel_sizes if kernel_sizes else [ceil(4 * sigma + 1) for sigma in sigma_values]

        self.setup_kernels()
        self.out_projection = (nn.Conv2d(len(sigma_values) * self.in_channels, out_channels, 1)
                               if out_channels not in (None, MISSING) else nn.Identity())

    @staticmethod
    def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size).float() - (size // 2)
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def setup_kernels(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NaiveGaussianFilterBank(BaseGaussianFilterBank):
    def setup_kernels(self):
        max_kernel_size = max(self.kernel_sizes)
        kernels = []

        for kernel_size, sigma in zip(self.kernel_sizes, self.sigma_values):
            if sigma > 0:
                kernel_1D = self.gaussian_kernel(kernel_size, sigma)
                kernel_2D = kernel_1D[:, None] * kernel_1D[None, :]
            else:
                kernel_2D = torch.tensor([[0.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0],
                                          [0.0, 0.0, 0.0]])
                kernel_size = 3

            padding = max_kernel_size - kernel_size
            pad_left = padding // 2
            pad_right = padding - pad_left
            kernel_2D = F.pad(kernel_2D, (pad_left, pad_right, pad_left, pad_right))
            kernels.append(kernel_2D[None, None, :, :])

        all_kernels = torch.cat(kernels, dim=0)
        self.register_buffer('all_kernels', all_kernels.repeat(self.in_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.all_kernels, groups=self.in_channels, padding="same")
        return self.out_projection(x)


class SeparableGaussianFilterBank(BaseGaussianFilterBank):
    def setup_kernels(self):
        max_kernel_size = max(self.kernel_sizes)
        kernels = []

        for kernel_size, sigma in zip(self.kernel_sizes, self.sigma_values):
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

        kernels = torch.stack(kernels)
        self.register_buffer('kernels_h', kernels.unsqueeze(1).unsqueeze(-1).repeat_interleave(self.in_channels, dim=0))
        self.register_buffer('kernels_v', kernels.unsqueeze(1).unsqueeze(2).repeat_interleave(self.in_channels, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_filtered = F.conv2d(x, self.kernels_h, groups=self.in_channels, padding='same')
        v_filtered = F.conv2d(h_filtered, self.kernels_v, groups=self.in_channels * len(self.sigma_values),
                              padding='same')
        return self.out_projection(v_filtered)


class FFTGaussianFilterBank(BaseGaussianFilterBank):
    def setup_kernels(self):
        pass

    def get_gaussian_freq(self, sigma: torch.Tensor, shape: tuple) -> torch.Tensor:
        rows, cols = shape[-2:]
        cols_rfft = cols // 2 + 1  # Only need half + 1 frequencies for real input
        center_row = rows // 2

        y = torch.arange(-center_row, rows - center_row, device=sigma.device)
        x = torch.arange(cols_rfft, device=sigma.device)  # Only positive frequencies
        Y, X = torch.meshgrid(y, x, indexing='ij')

        sigma = sigma.view(-1, 1, 1)
        return torch.where(
            sigma == 0,
            torch.ones((1, rows, cols_rfft), device=sigma.device),
            torch.exp(-2 * (torch.pi ** 2) * sigma ** 2 * (X ** 2 + Y ** 2) / (rows * cols))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma_tensor = torch.tensor(self.sigma_values, device=x.device)
        H = self.get_gaussian_freq(sigma_tensor, x.shape)

        x_freq = torch.fft.rfft2(x)
        x_freq = torch.fft.fftshift(x_freq, dim=(-2,))  # Only shift rows

        filtered_freq = x_freq.unsqueeze(1) * H.unsqueeze(1).unsqueeze(0)
        filtered_freq = torch.fft.ifftshift(filtered_freq, dim=(-2,))
        filtered = torch.fft.irfft2(filtered_freq, s=x.shape[-2:]).real

        return self.out_projection(einops.rearrange(filtered, 'b f c h w -> b (f c) h w'))


# Default implementation
GaussianFilterBank = SeparableGaussianFilterBank


def _getImg():
    import skimage
    image = skimage.data.camera()

    # Convert to PyTorch tensor and add batch and channel dimensions
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return image_tensor

def _getParticle(): #TODO: use a small set of data for test
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
    GaussianFilters = FFTGaussianFilterBank #SeparableGaussianFilterBank #FFTGaussianFilterBank #NaiveGaussianFilterBank
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
    # _visual_test()
    run_benchmark()

