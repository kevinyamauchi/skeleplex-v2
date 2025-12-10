import numpy as np
import pytest
import torch

from skeleplex.utils import FlipX, FlipY, FlipZ, TTAWrapper


class PassthroughModel(torch.nn.Module):
    """Mock model that returns input unchanged."""

    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Return input unchanged."""
        return x


@pytest.fixture
def passthrough_model():
    """Create a passthrough model."""
    return PassthroughModel()


@pytest.fixture
def augmentations():
    """Create list of flip augmentations."""
    return [FlipZ(), FlipY(), FlipX()]


def test_tta_wrapper_3d_torch_no_softmax(passthrough_model, augmentations):
    """Test TTAWrapper with 3D torch tensor without softmax."""
    np.random.seed(42)
    input_array = torch.from_numpy(np.random.rand(20, 20, 20).astype(np.float32))

    wrapper = TTAWrapper(
        model=passthrough_model,
        augmentations=augmentations,
        apply_softmax=False,
    )

    output = wrapper(input_array)

    assert isinstance(output, torch.Tensor)
    np.testing.assert_allclose(
        output.numpy(), input_array.numpy(), rtol=1e-6, atol=1e-6
    )


def test_tta_wrapper_4d_torch_no_softmax(passthrough_model, augmentations):
    """Test TTAWrapper with 4D torch tensor without softmax."""
    np.random.seed(42)
    input_array = torch.from_numpy(np.random.rand(2, 20, 20, 20).astype(np.float32))

    wrapper = TTAWrapper(
        model=passthrough_model,
        augmentations=augmentations,
        apply_softmax=False,
    )

    output = wrapper(input_array)

    assert isinstance(output, torch.Tensor)
    np.testing.assert_allclose(
        output.numpy(), input_array.numpy(), rtol=1e-6, atol=1e-6
    )


def test_tta_wrapper_5d_torch_no_softmax(passthrough_model, augmentations):
    """Test TTAWrapper with 5D torch tensor without softmax."""
    np.random.seed(42)
    input_array = torch.from_numpy(np.random.rand(1, 2, 20, 20, 20).astype(np.float32))

    wrapper = TTAWrapper(
        model=passthrough_model,
        augmentations=augmentations,
        apply_softmax=False,
    )

    output = wrapper(input_array)

    assert isinstance(output, torch.Tensor)
    np.testing.assert_allclose(
        output.numpy(), input_array.numpy(), rtol=1e-6, atol=1e-6
    )


def test_tta_wrapper_4d_torch_with_softmax(passthrough_model, augmentations):
    """Test TTAWrapper with 4D torch tensor with softmax."""
    np.random.seed(42)
    input_array = torch.from_numpy(np.random.rand(2, 20, 20, 20).astype(np.float32))

    wrapper = TTAWrapper(
        model=passthrough_model,
        augmentations=augmentations,
        apply_softmax=True,
    )

    output = wrapper(input_array)

    assert isinstance(output, torch.Tensor)
    expected = torch.softmax(input_array, dim=0)
    np.testing.assert_allclose(output.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


def test_tta_wrapper_3d_numpy_no_softmax(passthrough_model, augmentations):
    """Test TTAWrapper with 3D numpy array without softmax."""
    np.random.seed(42)
    input_array = np.random.rand(20, 20, 20).astype(np.float32)

    wrapper = TTAWrapper(
        model=passthrough_model,
        augmentations=augmentations,
        apply_softmax=False,
    )

    output = wrapper(input_array)

    assert isinstance(output, np.ndarray)
    np.testing.assert_allclose(output, input_array, rtol=1e-6, atol=1e-6)
