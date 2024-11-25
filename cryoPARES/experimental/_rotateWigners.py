import torch
from e3nn.o3 import wigner_D, angles_to_matrix, matrix_to_angles
import numpy as np


def get_wigner_coefficients(angles, lmax):
    """Get Wigner-D coefficients for given angles and maximum degree"""
    return [(2 * l + 1) ** 0.5 * wigner_D(l, *angles) for l in range(lmax + 1)]


def rotate_signal_coefficients(coeffs, rotation_angles, lmax):
    rotated_coeffs = []
    for l in range(lmax + 1):
        D = wigner_D(l, *rotation_angles)
        rotated = D @ coeffs[l]
        rotated_coeffs.append(rotated)
    return rotated_coeffs


def compose_rotations(angles1, angles2):
    """Compose two rotations given as ZYZ Euler angles"""
    R1 = angles_to_matrix(*angles1)
    R2 = angles_to_matrix(*angles2)
    R_composed = R2 @ R1
    return matrix_to_angles(R_composed)


def test_rotation_properties():
    lmax = 3
    atol = 1e-3

    # Test angles
    angles_90y = torch.tensor([0.0, np.pi / 2, 0.0])
    angles_180y = torch.tensor([0.0, np.pi, 0.0])
    angles_45y = torch.tensor([0.0, np.pi / 4, 0.0])
    angles_30y = torch.tensor([0.0, np.pi / 6, 0.0])
    angles_360y = torch.tensor([0.0, 2 * np.pi, 0.0])

    # Test 1: Direct coefficient computation vs rotation from different starting points
    initial_angles_list = [
        torch.tensor([np.pi / 6, np.pi / 4, np.pi / 3]),
        torch.tensor([np.pi / 2, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, np.pi / 2]),
        torch.rand(3) * 2 * np.pi,
    ]

    for initial_angles in initial_angles_list:
        for rotation_angles in [angles_90y, angles_180y, angles_45y]:
            init_coeffs = get_wigner_coefficients(initial_angles, lmax)

            # Method 1: Direct computation of final position
            final_angles = compose_rotations(initial_angles, rotation_angles)
            direct_coeffs = get_wigner_coefficients(final_angles, lmax)

            # Method 2: Rotate initial coefficients
            rotated_coeffs = rotate_signal_coefficients(init_coeffs, rotation_angles, lmax)

            for l in range(lmax + 1):
                assert torch.allclose(direct_coeffs[l], rotated_coeffs[l], atol=atol), \
                    f"Failed for initial={initial_angles * 180 / np.pi}°, rotation={rotation_angles * 180 / np.pi}°, l={l}"
    print("✓ Direct vs Rotated coefficients test passed for all starting points")

    # Test 2: Multiple rotations composition
    init_coeffs = get_wigner_coefficients(torch.zeros(3), lmax)

    # Two 45° = 90°
    rot_45 = rotate_signal_coefficients(init_coeffs, angles_45y, lmax)
    rot_45_twice = rotate_signal_coefficients(rot_45, angles_45y, lmax)
    rot_90 = rotate_signal_coefficients(init_coeffs, angles_90y, lmax)

    for l in range(lmax + 1):
        assert torch.allclose(rot_45_twice[l], rot_90[l], atol=atol)
    print("✓ 45°+45° = 90° test passed")

    # Three 30° = 90°
    rot_30 = rotate_signal_coefficients(init_coeffs, angles_30y, lmax)
    rot_30_twice = rotate_signal_coefficients(rot_30, angles_30y, lmax)
    rot_30_thrice = rotate_signal_coefficients(rot_30_twice, angles_30y, lmax)

    for l in range(lmax + 1):
        assert torch.allclose(rot_30_thrice[l], rot_90[l], atol=atol)
    print("✓ 30°+30°+30° = 90° test passed")

    # Test 3: Identity rotation from different starting points
    for initial_angles in initial_angles_list:
        init_coeffs = get_wigner_coefficients(initial_angles, lmax)
        rot_360 = rotate_signal_coefficients(init_coeffs, angles_360y, lmax)
        for l in range(lmax + 1):
            assert torch.allclose(rot_360[l], init_coeffs[l], atol=atol)
    print("✓ 360° rotation test passed for all starting points")

    # Test 4: Random angle rotations with proper composition
    for initial_angles in initial_angles_list:
        init_coeffs = get_wigner_coefficients(initial_angles, lmax)
        for _ in range(3):
            random_angles = torch.rand(3) * 2 * np.pi
            composed_angles = compose_rotations(initial_angles, random_angles)
            rot_coeffs = get_wigner_coefficients(composed_angles, lmax)
            rotated_coeffs = rotate_signal_coefficients(init_coeffs, random_angles, lmax)

            for l in range(lmax + 1):
                assert torch.allclose(rot_coeffs[l], rotated_coeffs[l], atol=atol)
    print("✓ Random angle rotations test passed")


if __name__ == "__main__":
    test_rotation_properties()