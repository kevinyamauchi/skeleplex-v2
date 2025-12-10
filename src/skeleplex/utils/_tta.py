"""Test time augmentation wrapper for PyTorch models."""

import numpy as np
import torch

from skeleplex.utils._tta_augmentations import Augmentation


class TTAWrapper:
    """Wrapper for test time augmentation inference with PyTorch models.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to use for prediction.
    augmentations : list[Augmentation]
        List of augmentation objects to apply during TTA.
    apply_softmax : bool, optional
        Whether to apply softmax along channel dimension (dim=1)
        before averaging predictions. Default is False.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        augmentations: list[Augmentation],
        apply_softmax: bool = False,
    ):
        self.model = model
        self.augmentations = augmentations
        self.apply_softmax = apply_softmax
        self.n_augmentations = len(augmentations)

        self.device = next(model.parameters()).device

        self.model.eval()

    def _coerce_input(
        self, array: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Coerce input array to 5D BCZYX tensor.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input array in ZYX (3D) or CZYX (4D) format.

        Returns
        -------
        tensor_5d : torch.Tensor
            Input tensor in BCZYX format on model device.
        metadata : dict
            Metadata for output coercion containing:
            - original_type: "numpy" or "torch"
            - original_shape: original array shape
            - original_device: original device (for torch tensors)
        """
        metadata = {}

        if isinstance(array, np.ndarray):
            metadata["original_type"] = "numpy"
            metadata["original_shape"] = array.shape
            tensor = torch.from_numpy(array)
        else:
            metadata["original_type"] = "torch"
            metadata["original_shape"] = array.shape
            metadata["original_device"] = array.device
            tensor = array

        if tensor.ndim == 3:
            tensor_5d = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 4:
            tensor_5d = tensor.unsqueeze(0)
        elif tensor.ndim == 5:
            tensor_5d = tensor
        else:
            raise ValueError(
                f"Input must be 3D (ZYX), 4D (CZYX), or 5D (BCZYX), "
                f"got {tensor.ndim}D"
            )

        tensor_5d = tensor_5d.to(self.device)

        return tensor_5d, metadata

    def _coerce_output(
        self, tensor_5d: torch.Tensor, metadata: dict
    ) -> np.ndarray | torch.Tensor:
        """Coerce output tensor back to original format.

        Parameters
        ----------
        tensor_5d : torch.Tensor
            Output tensor in BCZYX format.
        metadata : dict
            Metadata from input coercion containing original format info.

        Returns
        -------
        array : np.ndarray or torch.Tensor
            Output array in original format and shape.
        """
        original_ndim = len(metadata["original_shape"])

        if original_ndim == 3:
            tensor = tensor_5d.squeeze(0).squeeze(0)
        elif original_ndim == 4:
            tensor = tensor_5d.squeeze(0)
        elif original_ndim == 5:
            tensor = tensor_5d
        else:
            raise ValueError(f"Unexpected original shape: {original_ndim}D")

        if metadata["original_type"] == "numpy":
            return tensor.cpu().numpy()
        else:
            original_device = metadata["original_device"]
            return tensor.to(original_device)

    def _predict_with_tta(self, tensor: torch.Tensor) -> torch.Tensor:
        """Make prediction with test time augmentation.

        Predictions are accumulated and averaged to save memory.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor in BCZYX format on model device.

        Returns
        -------
        torch.Tensor
            Averaged prediction in BCZYX format.
        """
        accumulated_pred = None

        for augmentation in self.augmentations:
            augmented = augmentation(tensor)

            with torch.no_grad():
                pred = self.model(augmented)

            de_augmented_pred = augmentation.inverse(pred)

            if self.apply_softmax:
                de_augmented_pred = torch.softmax(de_augmented_pred, dim=1)

            if accumulated_pred is None:
                accumulated_pred = de_augmented_pred
            else:
                accumulated_pred = accumulated_pred + de_augmented_pred

        return accumulated_pred / self.n_augmentations

    def __call__(self, array: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Perform TTA inference on input array.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input array in ZYX (3D) or CZYX (4D) format.

        Returns
        -------
        result : np.ndarray or torch.Tensor
            Prediction in same format as input.
        """
        tensor_5d, metadata = self._coerce_input(array)

        prediction = self._predict_with_tta(tensor_5d)

        return self._coerce_output(prediction, metadata)
