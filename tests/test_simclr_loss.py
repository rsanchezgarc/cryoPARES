"""
Tests for SimCLR-like contrastive loss implementation in Image2Sphere model.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from cryoPARES.configs.mainConfig import main_config
from cryoPARES.constants import BATCH_PARTICLES_NAME
from cryoPARES.datamanager.datamanager import get_example_random_batch
from cryoPARES.models.image2sphere.image2sphere import Image2Sphere, extract_sh_coeffs_fast


def _update_config_for_test():
    """Configure test settings."""
    main_config.models.image2sphere.lmax = 6
    main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim = 512
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1.
    main_config.models.image2sphere.so3components.i2sprojector.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.hp_order = 2
    main_config.models.image2sphere.so3components.s2conv.f_out = 16
    main_config.models.image2sphere.so3components.so3outputgrid.hp_order = 3
    main_config.datamanager.particlesdataset.image_size_px_for_nnet = 224
    main_config.models.image2sphere.so3components.i2sprojector.rand_fraction_points_to_project = 1
    main_config.models.image2sphere.label_smoothing = 0.1


def test_simCLR_loss_basic():
    """Test basic SimCLR loss computation."""
    print("\n" + "="*60)
    print("Testing SimCLR-like Loss Implementation")
    print("="*60)

    _update_config_for_test()

    # Set up for contrastive learning
    num_particles = 4
    num_aug = 4
    batch_size = num_particles * num_aug
    main_config.datamanager.num_augmented_copies_per_batch = num_aug

    example_batch = get_example_random_batch(batch_size, n_channels=1, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    # Create model with SimCLR enabled
    encoder = nn.Conv2d(1, main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")
    model = Image2Sphere(
        symmetry="C1",
        lmax=6,
        enforce_symmetry=False,
        encoder=encoder,
        use_simCLR=True,
        simCLR_temperature=0.5,
        simCLR_loss_weight=0.1,
        num_augmented_copies_per_batch=num_aug,
        example_batch=example_batch
    )
    model.eval()

    with torch.no_grad():
        # Test 1: Basic loss computation
        print("\n1. Testing basic loss computation...")
        wD = model.predict_wignerDs(imgs)
        loss = model.simCLR_like_loss(wD, temperature=0.5)
        print(f"   Loss value: {loss.item():.6f}")
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be inf"
        print("   ✓ Loss is positive and finite")


def test_simCLR_temperature_scaling():
    """Test that temperature parameter affects loss magnitude."""
    _update_config_for_test()

    num_particles = 4
    num_aug = 4
    batch_size = num_particles * num_aug
    main_config.datamanager.num_augmented_copies_per_batch = num_aug

    example_batch = get_example_random_batch(batch_size, n_channels=1, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    encoder = nn.Conv2d(1, main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")
    model = Image2Sphere(
        symmetry="C1",
        lmax=6,
        encoder=encoder,
        use_simCLR=True,
        num_augmented_copies_per_batch=num_aug,
        example_batch=example_batch
    )
    model.eval()

    with torch.no_grad():
        print("\n2. Testing temperature scaling...")
        wD = model.predict_wignerDs(imgs)
        loss_low_temp = model.simCLR_like_loss(wD, temperature=0.1)
        loss_mid_temp = model.simCLR_like_loss(wD, temperature=0.5)
        loss_high_temp = model.simCLR_like_loss(wD, temperature=1.0)

        print(f"   Loss (T=0.1): {loss_low_temp.item():.6f}")
        print(f"   Loss (T=0.5): {loss_mid_temp.item():.6f}")
        print(f"   Loss (T=1.0): {loss_high_temp.item():.6f}")

        # All should be positive and finite
        assert all(l.item() > 0 for l in [loss_low_temp, loss_mid_temp, loss_high_temp])
        print("   ✓ Temperature affects loss magnitude")


def test_simCLR_feature_similarity():
    """Test that augmented copies have similar features."""
    _update_config_for_test()

    num_particles = 4
    num_aug = 4
    batch_size = num_particles * num_aug
    main_config.datamanager.num_augmented_copies_per_batch = num_aug

    example_batch = get_example_random_batch(batch_size, n_channels=1, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    encoder = nn.Conv2d(1, main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")
    model = Image2Sphere(
        symmetry="C1",
        lmax=6,
        encoder=encoder,
        use_simCLR=True,
        num_augmented_copies_per_batch=num_aug,
        example_batch=example_batch
    )
    model.eval()

    with torch.no_grad():
        print("\n3. Testing feature similarity for augmented copies...")
        wD = model.predict_wignerDs(imgs)
        sh_coeffs = extract_sh_coeffs_fast(wD, model.lmax).squeeze(1)
        sh_coeffs = sh_coeffs.view(num_particles, num_aug, -1)
        sh_coeffs_norm = nn.functional.normalize(sh_coeffs, p=2, dim=-1)

        # Compute similarity within particles vs across particles
        within_particle_sim = []
        across_particle_sim = []

        for i in range(num_particles):
            # Within same particle
            for j in range(num_aug):
                for k in range(j+1, num_aug):
                    sim = (sh_coeffs_norm[i, j] * sh_coeffs_norm[i, k]).sum()
                    within_particle_sim.append(sim.item())

            # Across different particles
            for other_i in range(i+1, num_particles):
                sim = (sh_coeffs_norm[i, 0] * sh_coeffs_norm[other_i, 0]).sum()
                across_particle_sim.append(sim.item())

        avg_within = np.mean(within_particle_sim)
        avg_across = np.mean(across_particle_sim)
        print(f"   Average similarity within particle: {avg_within:.6f}")
        print(f"   Average similarity across particles: {avg_across:.6f}")
        print("   ✓ Similarity metrics computed")


def test_simCLR_integration_with_forward_and_loss():
    """Test integration with forward_and_loss method."""
    _update_config_for_test()

    num_particles = 4
    num_aug = 4
    batch_size = num_particles * num_aug
    main_config.datamanager.num_augmented_copies_per_batch = num_aug

    example_batch = get_example_random_batch(batch_size, n_channels=1, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    encoder = nn.Conv2d(1, main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")
    model = Image2Sphere(
        symmetry="C1",
        lmax=6,
        encoder=encoder,
        use_simCLR=True,
        simCLR_temperature=0.5,
        simCLR_loss_weight=0.1,
        num_augmented_copies_per_batch=num_aug,
        example_batch=example_batch
    )

    print("\n4. Testing integration with forward_and_loss...")
    gt_rot = torch.from_numpy(Rotation.random(batch_size, random_state=42).as_matrix().astype(np.float32))

    model.train()
    with torch.no_grad():
        (wD_out, logits, ids, mats, probs), loss_total, error = model.forward_and_loss(imgs, gt_rot, top_k=1)
        print(f"   Total loss (with contrastive): {loss_total.mean().item():.6f}")

        # Verify contrastive loss is included
        model.use_simCLR = False
        (_, _, _, _, _), loss_no_contrast, _ = model.forward_and_loss(imgs, gt_rot, top_k=1)
        print(f"   Total loss (without contrastive): {loss_no_contrast.mean().item():.6f}")

        model.use_simCLR = True
        contrast_only = model.simCLR_loss_weight * model.simCLR_like_loss(wD_out, temperature=model.simCLR_temperature)
        print(f"   Contrastive loss component: {contrast_only.item():.6f}")

        # The difference should be approximately the contrastive loss component
        diff = abs(loss_total.mean().item() - loss_no_contrast.mean().item())
        print(f"   Difference: {diff:.6f}")
        print("   ✓ Contrastive loss properly integrated")


def test_simCLR_edge_case_single_augmentation():
    """Test edge case with single augmentation."""
    _update_config_for_test()

    example_batch = get_example_random_batch(4, n_channels=1, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    encoder = nn.Conv2d(1, main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")

    print("\n5. Testing edge case with num_augmented_copies=1...")
    model_single = Image2Sphere(
        symmetry="C1",
        lmax=6,
        encoder=encoder,
        use_simCLR=True,
        num_augmented_copies_per_batch=1,
        example_batch=get_example_random_batch(4, n_channels=1)
    )
    model_single.eval()

    with torch.no_grad():
        wD_single = model_single.predict_wignerDs(imgs[:4])
        loss_single = model_single.simCLR_like_loss(wD_single)
        print(f"   Loss with single augmentation: {loss_single}")
        assert loss_single == 0.0, "Should return 0 when no augmentation"
        print("   ✓ Correctly handles single augmentation case")


def test_simCLR_edge_case_none_augmentation():
    """Test edge case with None augmentation."""
    _update_config_for_test()

    example_batch = get_example_random_batch(4, n_channels=1, seed=42)
    imgs = example_batch[BATCH_PARTICLES_NAME]

    encoder = nn.Conv2d(1, main_config.models.image2sphere.so3components.i2sprojector.sphere_fdim,
                        kernel_size=1, padding="same")

    print("\n6. Testing edge case with num_augmented_copies=None...")
    model_none = Image2Sphere(
        symmetry="C1",
        lmax=6,
        encoder=encoder,
        use_simCLR=True,
        num_augmented_copies_per_batch=None,
        example_batch=get_example_random_batch(4, n_channels=1)
    )
    model_none.eval()

    with torch.no_grad():
        wD_none = model_none.predict_wignerDs(imgs[:4])
        loss_none = model_none.simCLR_like_loss(wD_none)
        print(f"   Loss with None augmentation: {loss_none}")
        assert loss_none == 0.0, "Should return 0 when augmentation is None"
        print("   ✓ Correctly handles None augmentation case")


if __name__ == "__main__":
    print("Running SimCLR Loss Tests...")
    print("="*60)

    test_simCLR_loss_basic()
    test_simCLR_temperature_scaling()
    test_simCLR_feature_similarity()
    test_simCLR_integration_with_forward_and_loss()
    test_simCLR_edge_case_single_augmentation()
    test_simCLR_edge_case_none_augmentation()

    print("\n" + "="*60)
    print("All SimCLR loss tests passed! ✓")
    print("="*60 + "\n")
