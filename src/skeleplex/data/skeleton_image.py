"""Example skeleton images."""

import numpy as np
from skimage.draw import line


def simple_t() -> np.ndarray:
    """Make an image with a simple T skeleton.

    Returns
    -------
    np.ndarray
        A binary image with a simple T skeleton.
        The image has shape (20, 20, 20)
    """
    # node coordinates for each branch
    branch_coordinates = [
        [(10, 5), (10, 10)],
        [(10, 10), (10, 15)],
        [(10, 10), (15, 10)],
    ]

    # draw the image
    image = np.zeros((20, 20, 20), dtype=bool)

    for branch in branch_coordinates:
        rr, cc = line(*branch[0], *branch[1])
        image[10, rr, cc] = 1

    return image


def big_t() -> np.ndarray:
    """Make an image with a big T skeleton.

    Returns
    -------
    np.ndarray
        A binary image with a big T skeleton.
        The image has shape (100, 100, 100)
    """
    # node coordinates for each branch
    branch_coordinates = [
        [(50, 25), (50, 50)],
        [(50, 50), (50, 75)],
        [(50, 50), (75, 50)],
    ]

    # draw the image
    image = np.zeros((100, 100, 100), dtype=bool)

    for branch in branch_coordinates:
        rr, cc = line(*branch[0], *branch[1])
        image[50, rr, cc] = 1

    return image


def simple_bifurcating_branches() -> np.ndarray:
    """Create a skeleton image with a simple bifurcating tree."""
    # node coordinates for each branch
    branch_coordinates = [
        [(20, 100), (40, 100)],
        [(40, 100), (60, 80)],
        [(40, 100), (60, 120)],
        [(60, 80), (75, 65)],
        [(60, 80), (75, 95)],
        [(60, 120), (75, 105)],
        [(60, 120), (75, 135)],
    ]

    # draw the image
    image = np.zeros((200, 200, 200), dtype=bool)

    for branch in branch_coordinates:
        rr, cc = line(*branch[0], *branch[1])
        image[100, rr, cc] = 1

    return image
