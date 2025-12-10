"""Augmentation classes for test time augmentation."""

from abc import ABC, abstractmethod

import torch


class Augmentation(ABC):
    """Base class for test time augmentations."""

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        pass

    @abstractmethod
    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        pass


class Identity(Augmentation):
    """No transformation applied."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return tensor

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return tensor


class FlipZ(Augmentation):
    """Flip along Z axis (dim=2 in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.flip(tensor, dims=[2])

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.flip(tensor, dims=[2])


class FlipY(Augmentation):
    """Flip along Y axis (dim=3 in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.flip(tensor, dims=[3])

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.flip(tensor, dims=[3])


class FlipX(Augmentation):
    """Flip along X axis (dim=4 in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.flip(tensor, dims=[4])

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.flip(tensor, dims=[4])


class Rot90ZY(Augmentation):
    """Rotate 90 degrees in ZY plane (dims=(2,3) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=1, dims=(2, 3))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-1, dims=(2, 3))


class Rot180ZY(Augmentation):
    """Rotate 180 degrees in ZY plane (dims=(2,3) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=2, dims=(2, 3))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-2, dims=(2, 3))


class Rot270ZY(Augmentation):
    """Rotate 270 degrees in ZY plane (dims=(2,3) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=3, dims=(2, 3))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-3, dims=(2, 3))


class Rot90ZX(Augmentation):
    """Rotate 90 degrees in ZX plane (dims=(2,4) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=1, dims=(2, 4))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-1, dims=(2, 4))


class Rot180ZX(Augmentation):
    """Rotate 180 degrees in ZX plane (dims=(2,4) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=2, dims=(2, 4))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-2, dims=(2, 4))


class Rot270ZX(Augmentation):
    """Rotate 270 degrees in ZX plane (dims=(2,4) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=3, dims=(2, 4))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-3, dims=(2, 4))


class Rot90YX(Augmentation):
    """Rotate 90 degrees in YX plane (dims=(3,4) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=1, dims=(3, 4))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-1, dims=(3, 4))


class Rot180YX(Augmentation):
    """Rotate 180 degrees in YX plane (dims=(3,4) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=2, dims=(3, 4))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-2, dims=(3, 4))


class Rot270YX(Augmentation):
    """Rotate 270 degrees in YX plane (dims=(3,4) in BCZYX)."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply forward augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            Augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=3, dims=(3, 4))

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply inverse augmentation transform.

        Parameters
        ----------
        tensor : torch.Tensor
            Predicted tensor in BCZYX format.

        Returns
        -------
        torch.Tensor
            De-augmented tensor in BCZYX format.
        """
        return torch.rot90(tensor, k=-3, dims=(3, 4))
