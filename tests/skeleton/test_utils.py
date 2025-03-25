import numpy as np
import pytest

from skeleplex.skeleton._utils import make_image_5d


@pytest.mark.parametrize(
    "image",
    [
        np.ones((5, 5, 5)),
        np.ones((5, 5, 5, 5)),
        np.ones((5, 5, 5, 5, 5)),
    ],
)
def test_make_image_5d(image):
    """Test making an image 5D."""
    result = make_image_5d(image)
    assert result.ndim == 5


def test_make_image_5d_invalid_image():
    """Test making an image 5D with an invalid image."""
    with pytest.raises(ValueError):
        # not compatible with 2D image.
        make_image_5d(np.ones((5, 5)))
