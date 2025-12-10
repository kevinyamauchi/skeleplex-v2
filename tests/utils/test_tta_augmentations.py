import numpy as np
import pytest
import torch

from skeleplex.utils import (
    FlipX,
    FlipY,
    FlipZ,
    Identity,
    Rot90YX,
    Rot90ZX,
    Rot90ZY,
    Rot180YX,
    Rot180ZX,
    Rot180ZY,
    Rot270YX,
    Rot270ZX,
    Rot270ZY,
)


@pytest.fixture
def test_array():
    """Create a test array with known values."""
    np.random.seed(42)
    array = np.random.rand(1, 1, 20, 20, 20).astype(np.float32)
    return torch.from_numpy(array)


def test_identity_forward_inverse(test_array):
    """Test Identity augmentation forward and inverse."""
    aug = Identity()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    np.testing.assert_allclose(forward.numpy(), test_array.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_flip_z_forward_inverse(test_array):
    """Test FlipZ augmentation forward and inverse."""
    aug = FlipZ()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.flip(test_array, dims=[2])
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_flip_y_forward_inverse(test_array):
    """Test FlipY augmentation forward and inverse."""
    aug = FlipY()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.flip(test_array, dims=[3])
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_flip_x_forward_inverse(test_array):
    """Test FlipX augmentation forward and inverse."""
    aug = FlipX()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.flip(test_array, dims=[4])
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot90_zy_forward_inverse(test_array):
    """Test Rot90ZY augmentation forward and inverse."""
    aug = Rot90ZY()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=1, dims=(2, 3))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot180_zy_forward_inverse(test_array):
    """Test Rot180ZY augmentation forward and inverse."""
    aug = Rot180ZY()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=2, dims=(2, 3))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot270_zy_forward_inverse(test_array):
    """Test Rot270ZY augmentation forward and inverse."""
    aug = Rot270ZY()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=3, dims=(2, 3))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot90_zx_forward_inverse(test_array):
    """Test Rot90ZX augmentation forward and inverse."""
    aug = Rot90ZX()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=1, dims=(2, 4))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot180_zx_forward_inverse(test_array):
    """Test Rot180ZX augmentation forward and inverse."""
    aug = Rot180ZX()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=2, dims=(2, 4))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot270_zx_forward_inverse(test_array):
    """Test Rot270ZX augmentation forward and inverse."""
    aug = Rot270ZX()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=3, dims=(2, 4))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot90_yx_forward_inverse(test_array):
    """Test Rot90YX augmentation forward and inverse."""
    aug = Rot90YX()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=1, dims=(3, 4))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot180_yx_forward_inverse(test_array):
    """Test Rot180YX augmentation forward and inverse."""
    aug = Rot180YX()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=2, dims=(3, 4))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())


def test_rot270_yx_forward_inverse(test_array):
    """Test Rot270YX augmentation forward and inverse."""
    aug = Rot270YX()

    forward = aug(test_array)
    inverse = aug.inverse(forward)

    expected_forward = torch.rot90(test_array, k=3, dims=(3, 4))
    np.testing.assert_allclose(forward.numpy(), expected_forward.numpy())
    np.testing.assert_allclose(inverse.numpy(), test_array.numpy())
