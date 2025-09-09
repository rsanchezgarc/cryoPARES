import time

import numpy as np
import scipy
import torch



class RingManager(torch.nn.Module):
    def __init__(self, image_shape, num_rings=None,
                 rfft=False, fftshift=True):

        super().__init__()
        if fftshift == False:
            raise  NotImplementedError("Error, fftshift=False not implemented")

        self.image_shape = image_shape
        self.rfft = rfft
        # Adjust for real FFT
        if rfft:
            fft_size = (image_shape[0], image_shape[1] // 2 + 1)
            center = torch.tensor(fft_size) // 2
            center[...,-1] *= 0
        else:
            fft_size = image_shape
            center = torch.tensor(fft_size) // 2

        # Automatically determine the number of rings if not provided
        if num_rings is None:
            num_rings = min(image_shape) // 2
        self.num_rings = num_rings

        # Precompute masks
        Y, X = torch.meshgrid(torch.arange(fft_size[0]), torch.arange(fft_size[1]), indexing='ij')

        R = torch.sqrt((X - center[1])**2 + (Y - center[0])**2)

        masks = torch.stack([(R >= i) & (R < i + 1) for i in range(num_rings)], dim=0)
        self.register_buffer("masks_counts", masks.sum(dim=(-2,-1)))
        masks = masks.to_sparse_coo()
        self.register_buffer("masks", masks)


class RelionProb(torch.nn.Module):

    def __init__(self, image_shape, num_rings=None,
                 rfft=False, fftshift=True):
        super().__init__()
        if fftshift == False:
            raise  NotImplementedError("Error, fftshift=False not implemented")

        self.image_shape = image_shape
        self.rfft = rfft
        # Adjust for real FFT
        if rfft:
            fft_size = (image_shape[0], image_shape[1] // 2 + 1)
            center = torch.tensor(fft_size) // 2
            center[...,-1] *= 0
        else:
            fft_size = image_shape
            center = torch.tensor(fft_size) // 2

        # Automatically determine the number of rings if not provided
        if num_rings is None:
            num_rings = min(image_shape) // 2
        self.num_rings = num_rings

        coords = torch.stack(torch.meshgrid(torch.arange(image_shape[-2]),
                                            torch.arange(image_shape[-1]), indexing="ij"), -1).float()
        coords -= center

        # Compute radial distances
        radial_distances = torch.sqrt((coords ** 2).sum(-1))

        # Bin the distances
        distance_bins = torch.round(radial_distances).view(-1).long()

        self.register_buffer("distance_bins", distance_bins)
        self.register_buffer("valid_bins_mask", (distance_bins > 3) & (distance_bins < num_rings))

    def get_logprob(self, parts, projs): #TODO: This estimates the noise_sigma using only one pose. In relion, it is estimated using all the poses
        #parts is signal+noise; shifted_projs is only signal -> noise parts-signal
        noise = (parts - projs)
        batch_size = noise.shape[0]
        _distance_bins = self.distance_bins.unsqueeze(0).expand(batch_size, -1)
        flat_noise = noise.view(batch_size,-1)
        #TODO: apply this before: self.valid_bins_mask
        import torch_scatter
        from torch_scatter import scatter_mean, scatter_std
        # radial_avg = scatter_mean(flat_noise, _distance_bins, dim=-1)
        radial_var = scatter_std(flat_noise.abs(), _distance_bins, dim=-1)**2
        batch_arange = torch.arange(_distance_bins.shape[0], device=parts.device).unsqueeze(-1)
        _flat_var = radial_var[batch_arange, _distance_bins]

        exp_term = -0.5*(flat_noise.square().abs()/_flat_var) #.view(1, *self.image_shape)

        logprob = exp_term - torch.log(2*torch.pi*_flat_var)
        logprob = logprob[:, self.valid_bins_mask]
        logprob = logprob.sum(-1, keepdim=True)
        return logprob

class FourierRingCorrelation(RingManager):

    def compute_frc_from_real(self, image1, image2):
        # Convert images to PyTorch tensors
        img1 = torch.as_tensor(image1, dtype=torch.float32)
        img2 = torch.as_tensor(image2, dtype=torch.float32)

        if self.rfft:
            fimage1 = torch.fft.fftshift(torch.fft.rfft2(img1, dim=(-2,-1)), dim=(-2,))
            fimage2 = torch.fft.fftshift(torch.fft.rfft2(img2, dim=(-2,-1)), dim=(-2,))
        else:
            fimage1 = torch.fft.fftshift(torch.fft.fft2(img1, dim=(-2,-1)), dim=(-2,-1))
            fimage2 = torch.fft.fftshift(torch.fft.fft2(img2, dim=(-2,-1)), dim=(-2,-1))

        # plt.imshow(fimage1.abs().log()); plt.show()
        if len(fimage1.shape) < 3:
            fimage1 = fimage1.unsqueeze(0)
            fimage2 = fimage2.unsqueeze(0)
            return self.compute_frc(fimage1, fimage2).squeeze(0)
        else:
            return self.compute_frc(fimage1, fimage2)



    def compute_frc(self, fimage1, fimage2):
        # Apply masks and calculate FRC for all rings
        b, *s, l0, l1 = fimage1.shape
        if len(s) > 0:
            #Broadcast seems not to work between sparse and dense. Thus the following does not work
            # masked_F1 = fimage1.unsqueeze(1) * self.masks.unsqueeze(0) #torch.einsum("bij, rij -> brij", fimage1, self.masks)
            # masked_F2 = fimage2.unsqueeze(1) * self.masks.unsqueeze(0)
            #torch.einsum("...ij, ..rij -> ...rij", fimage1, masks)
            #The torch.stack hack works!
            # t = time.time()
            masks = torch.stack([self.masks for _ in range(np.prod(fimage1.shape[:-2]))], dim=0)

            masked_F1 = torch.einsum("...ij, ...rij -> ...rij", fimage1.reshape(b * np.prod(s), l0, l1), masks)
            masked_F2 = torch.einsum("...ij, ...rij -> ...rij", fimage2.reshape(b * np.prod(s), l0, l1), masks)

        else:
            masked_F1 = fimage1 * self.masks
            masked_F2 = fimage2 * self.masks


        numerator = torch.abs(torch.sum(masked_F1 * torch.conj(masked_F2), dim=(-2, -1)))
        denominator = torch.sqrt(torch.sum(torch.abs(masked_F1)**2, dim=(-2, -1)) * torch.sum(torch.abs(masked_F2)**2, dim=(-2, -1)))
        frc_curve = numerator.to_dense() / denominator.to_dense()
        frc_curve = torch.nan_to_num(frc_curve, nan=0.)

        if s:
            frc_curve = frc_curve.reshape(b, *s, -1)

        return frc_curve

    def forward(self, fimage1, fimage2):
        return self.compute_frc(fimage1, fimage2)



def euler_degs_diff(euler1, euler2):
  # Initialize the rotation objects from the tensors of euler angles
  rot1 = scipy.spatial.transform.Rotation.from_euler('ZYZ', euler1, degrees=True)
  rot2 = scipy.spatial.transform.Rotation.from_euler('ZYZ', euler2, degrees=True)

  # Invert the first rotation object
  rot1_inv = rot1.inv()

  # Multiply the inverted first rotation object with the second rotation object
  # This gives the relative rotation from euler1 to euler2
  rot_diff = rot1_inv * rot2

  # Convert the relative rotation object to a tensor of euler angles
  # euler_diff = rot_diff.as_euler('ZYZ', degrees=True)
  # return euler_diff

  rot_diff = np.rad2deg(rot_diff.magnitude())
  return rot_diff

def shifst_angs_diff(shiftsAngs1, shiftsAngs2):
    return np.linalg.norm(shiftsAngs2-shiftsAngs1, axis=-1)


def _test_fourier_ring_correlation():
    from matplotlib import pyplot as plt

    # n = 128
    # test_image = np.random.rand(n, n)
    from skimage.data import camera
    test_image = camera()/255.
    n = test_image.shape[1]

    frc_analyzer_auto = FourierRingCorrelation((n, n))

    # Scenario 2: Random Gaussian Images
    random_gaussian1 = np.random.normal(size=(n, n))
    random_gaussian2 = np.random.normal(size=(n, n))

    # Perform FRC analysis using the class with automatic ring computation
    frc_same_auto = frc_analyzer_auto.compute_frc_from_real(test_image, test_image)
    frc_random_auto = frc_analyzer_auto.compute_frc_from_real(random_gaussian1, random_gaussian2)

    noise_levels = .1 * np.array([0.0, 0.1, 0.5, 1., 2.])
    # frc_noisy_auto = [frc_analyzer_auto.compute_frc_from_real(test_image, test_image + np.random.normal(scale=noise_level, size=test_image.shape)) for noise_level in noise_levels]

    test_images = [(test_image, test_image + np.random.normal(scale=noise_level, size=test_image.shape)) for noise_level in noise_levels]
    in_test, out_test = zip(*test_images)
    # frc_noisy_auto = [x for x in frc_analyzer_auto.compute_frc_from_real(in_test, out_test)]
    frc_noisy_auto = [x for x in frc_analyzer_auto.compute_frc_from_real(np.array(in_test), np.array(out_test))]
    # frc_noisy_auto = [frc_analyzer_auto.compute_frc_from_real(in_test[i], out_test[i]) for i in range(len(in_test))]

    # Print FRC values at specific rings for the automatic ring computation
    for label, frc_curve in zip(["Same Image", "Random Gaussian Images"] + [f"Noise Level {nl}" for nl in noise_levels],
                                [frc_same_auto, frc_random_auto] + frc_noisy_auto):
        print(f"{label}: {frc_curve.mean()}")
        # for ring in range(n//2):
        #     print(f"  Ring {ring}: {frc_curve[ring]}")

    # Plotting the results of the automatic ring computation
    plt.figure(figsize=(12, 8))
    plt.plot(frc_same_auto.numpy(), label='Same Image (Auto Rings)', linestyle='-', marker='o')
    plt.plot(frc_random_auto.numpy(), label='Random Gaussian Images (Auto Rings)', linestyle='--', marker='x')
    for i, noise_level in enumerate(noise_levels):
        plt.plot(frc_noisy_auto[i].numpy(), label=f'Noise Level {noise_level:3f} (Auto Rings)', linestyle='-.', marker='^')
    plt.xlabel('Ring Number')
    plt.ylabel('FRC')
    plt.legend()
    plt.title('Fourier Ring Correlation (FRC) Analysis with PyTorch (Automatic Rings)')
    plt.show()

def _test_relion_prob():
    from matplotlib import pyplot as plt

    from skimage.data import camera
    test_image = camera()/255.
    test_image = torch.FloatTensor(test_image).unsqueeze(0)
    n = test_image.shape[1]

    relion_prob_analyser = RelionProb((n, n))
    random_gaussian1 = np.random.normal(size=(1, n, n), scale=0.5)

    # Perform FRC analysis using the class with automatic ring computation
    frc_same_auto = relion_prob_analyser.get_logprob(test_image, test_image + random_gaussian1)

    # noise_levels = .1 * np.array([0.0, 0.1, 0.5, 1., 2.])
    # # frc_noisy_auto = [relion_prob_analyser.get_prob(test_image, test_image + np.random.normal(scale=noise_level, size=test_image.shape)) for noise_level in noise_levels]
    #
    # test_images = [(test_image, test_image + np.random.normal(scale=noise_level, size=test_image.shape)) for noise_level in noise_levels]
    # in_test, out_test = zip(*test_images)
    # # frc_noisy_auto = [x for x in relion_prob_analyser.get_prob(in_test, out_test)]
    # frc_noisy_auto = [x for x in relion_prob_analyser.get_prob(np.array(in_test), np.array(out_test))]
    # # frc_noisy_auto = [relion_prob_analyser.get_prob(in_test[i], out_test[i]) for i in range(len(in_test))]
    #
    # # Print FRC values at specific rings for the automatic ring computation
    # for label, frc_curve in zip(["Same Image", "Random Gaussian Images"] + [f"Noise Level {nl}" for nl in noise_levels],
    #                             [frc_same_auto, frc_random_auto] + frc_noisy_auto):
    #     print(f"{label}: {frc_curve.mean()}")
    #     # for ring in range(n//2):
    #     #     print(f"  Ring {ring}: {frc_curve[ring]}")
    #
    # # Plotting the results of the automatic ring computation
    # plt.figure(figsize=(12, 8))
    # plt.plot(frc_same_auto.numpy(), label='Same Image (Auto Rings)', linestyle='-', marker='o')
    # plt.plot(frc_random_auto.numpy(), label='Random Gaussian Images (Auto Rings)', linestyle='--', marker='x')
    # for i, noise_level in enumerate(noise_levels):
    #     plt.plot(frc_noisy_auto[i].numpy(), label=f'Noise Level {noise_level:3f} (Auto Rings)', linestyle='-.', marker='^')
    # plt.xlabel('Ring Number')
    # plt.ylabel('FRC')
    # plt.legend()
    # plt.title('Fourier Ring Correlation (FRC) Analysis with PyTorch (Automatic Rings)')
    # plt.show()

    raise NotImplementedError

if __name__ == "__main__":
    # _test_fourier_ring_correlation()
    _test_relion_prob()

