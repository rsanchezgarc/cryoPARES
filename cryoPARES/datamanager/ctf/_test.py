import torch

from cryoPARES.datamanager.ctf import fft_ctf, rfft_ctf
from cryoPARES.datamanager.ctf.common import convert_fft_to_rfft


def _test_ctf_conversion(image_size, visualize):
    """Test FFT and RFFT CTF generation with conversions"""
    print(f"\nTesting size: {image_size}")
    print(f"Expected RFFT size: {image_size // 2 + 1 if image_size % 2 == 0 else (image_size + 1) // 2}")

    # Test parameters
    test_params = {
        'sampling_rate': 1.0,
        'dfu': 1e4,
        'dfv': 1e4,
        'dfang': 0,
        'volt': 300,
        'cs': 2.7,
        'w': 0.1,
        'phase_shift': 0,
        'bfactor': None,
        'device': None
    }

    # Generate CTFs using both methods
    ctf_fft = fft_ctf.compute_ctf(image_size, **test_params)
    ctf_rfft = rfft_ctf.compute_ctf(image_size, **test_params)

    # Print shapes for verification
    print(f"FFT CTF shape: {ctf_fft.shape}")
    print(f"RFFT CTF shape: {ctf_rfft.shape}")

    # Convert FFT CTF to RFFT format
    ctf_fft_converted = convert_fft_to_rfft(ctf_fft)
    print(f"Converted CTF shape: {ctf_fft_converted.shape}")

    # Compare
    diff = torch.abs(ctf_rfft - ctf_fft_converted)
    max_diff = torch.max(diff)
    print(f"Maximum difference between CTFs: {max_diff}")

    # Create sample data for verification
    test_data = torch.randn(image_size, image_size)

    # Verify that both give same results with real data
    fft_result = torch.fft.fft2(test_data)
    rfft_result = torch.fft.rfft2(test_data)

    print(f"FFT result shape: {fft_result.shape}")
    print(f"RFFT result shape: {rfft_result.shape}")

    if visualize:
        # Visualize for debugging
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(ctf_fft.cpu())
        axes[0, 0].set_title(f'Original FFT CTF ({image_size}x{image_size})')

        axes[0, 1].imshow(ctf_fft_converted.cpu())
        axes[0, 1].set_title(f'FFT CTF converted to RFFT ({ctf_fft_converted.shape})')

        axes[1, 0].imshow(ctf_rfft.cpu())
        axes[1, 0].set_title(f'Direct RFFT CTF ({ctf_rfft.shape})')

        axes[1, 1].imshow(diff.cpu())
        axes[1, 1].set_title('Difference')

        plt.tight_layout()
        plt.show()

    return ctf_fft, ctf_rfft, ctf_fft_converted

def _test_ctf_conversion_multiplesizes(visualize=True):
    for size in [32, 33, 64, 65]:
        ctf_fft, ctf_rfft, ctf_fft_converted = _test_ctf_conversion(size, visualize=visualize)


def _test_ctf_operations(image_size=64, visualize=True):
    """Test both corrupt and correct operations between FFT and RFFT implementations"""
    print(f"\nTesting CTF operations with image size: {image_size}")

    # Test parameters
    params = {
        'sampling_rate': 1.0,
        'dfu': 1e4,
        'dfv': 1e4,
        'dfang': 0,
        'volt': 300,
        'cs': 2.7,
        'w': 0.1,
        'phase_shift': 0,
        'bfactor': None
    }

    # Create a test image with some features
    x = torch.linspace(-1, 1, image_size)
    y = torch.linspace(-1, 1, image_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    R = torch.sqrt(X ** 2 + Y ** 2)
    test_image = torch.exp(-10 * R ** 2) + 0.5 * torch.exp(-5 * (X - 0.5) ** 2 - 5 * (Y - 0.5) ** 2)

    print("\nTesting corrupt_with_ctf...")
    # Test corruption
    ctf_fft, corrupted_fft = fft_ctf.corrupt_with_ctf(test_image, **params)
    ctf_rfft, corrupted_rfft = rfft_ctf.corrupt_with_ctf(test_image, **params)

    corrupt_diff = torch.abs(corrupted_fft - corrupted_rfft)
    max_corrupt_diff = torch.max(corrupt_diff)
    print(f"Maximum difference in corrupted images: {max_corrupt_diff}")

    print("\nTesting correct_ctf...")
    # Test correction with phase flip
    _, corrected_fft_pf = fft_ctf.correct_ctf(corrupted_fft, mode='phase_flip', **params)
    _, corrected_rfft_pf = rfft_ctf.correct_ctf(corrupted_rfft, mode='phase_flip', **params)

    pf_diff = torch.abs(corrected_fft_pf - corrected_rfft_pf)
    max_pf_diff = torch.max(pf_diff)
    print(f"Maximum difference in phase flip correction: {max_pf_diff}")

    # Test correction with Wiener filter
    _, corrected_fft_w = fft_ctf.correct_ctf(corrupted_fft, mode='wiener', wiener_parameter=0.1, **params)
    _, corrected_rfft_w = rfft_ctf.correct_ctf(corrupted_rfft, mode='wiener', wiener_parameter=0.1, **params)

    wiener_diff = torch.abs(corrected_fft_w - corrected_rfft_w)
    max_wiener_diff = torch.max(wiener_diff)
    print(f"Maximum difference in Wiener correction: {max_wiener_diff}")

    if visualize:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        # Original and corrupted images
        axes[0, 0].imshow(test_image.cpu())
        axes[0, 0].set_title('Original Image')

        axes[0, 1].imshow(corrupted_fft.cpu())
        axes[0, 1].set_title('FFT Corrupted')

        axes[0, 2].imshow(corrupt_diff.cpu())
        axes[0, 2].set_title('Corruption Difference')

        # Phase flip correction
        axes[1, 0].imshow(corrected_fft_pf.cpu())
        axes[1, 0].set_title('FFT Phase Flip')

        axes[1, 1].imshow(corrected_rfft_pf.cpu())
        axes[1, 1].set_title('RFFT Phase Flip')

        axes[1, 2].imshow(pf_diff.cpu())
        axes[1, 2].set_title('Phase Flip Difference')

        # Wiener correction
        axes[2, 0].imshow(corrected_fft_w.cpu())
        axes[2, 0].set_title('FFT Wiener')

        axes[2, 1].imshow(corrected_rfft_w.cpu())
        axes[2, 1].set_title('RFFT Wiener')

        axes[2, 2].imshow(wiener_diff.cpu())
        axes[2, 2].set_title('Wiener Difference')

        plt.tight_layout()
        plt.show()

    return {
        'corrupt_diff': max_corrupt_diff,
        'phase_flip_diff': max_pf_diff,
        'wiener_diff': max_wiener_diff
    }


def _test_multiple_sizes_ctf_effect():
    """Test operations with different image sizes"""
    sizes = [32, 33, 64, 65]
    results = {}

    for size in sizes:
        print(f"\n{'=' * 50}")
        print(f"Testing size: {size}")
        results[size] = _test_ctf_operations(size, visualize=True)

    # Print summary
    print("\nSummary of maximum differences:")
    print("Size | Corrupt | Phase Flip | Wiener")
    print("-" * 40)
    for size, diffs in results.items():
        print(f"{size:4d} | {diffs['corrupt_diff']:.2e} | {diffs['phase_flip_diff']:.2e} | {diffs['wiener_diff']:.2e}")



if __name__ == "__main__":
    _test_ctf_conversion_multiplesizes(visualize=False)
    _test_ctf_operations()