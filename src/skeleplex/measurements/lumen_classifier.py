import logging  # noqa
import os
import random
import shutil

import h5py
import numpy as np
import pytorch_lightning as pl
import skimage as ski
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
from torchvision import models

from monai.transforms import EnsureType
from PIL import Image
from qtpy.QtWidgets import QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget
from torchvision.models import ResNet50_Weights


from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomAffine,
    RandomApply,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from skeleplex.measurements.utils import grey2rgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SaveClassifiedSlices(QWidget):
    """Widget to save classified slices in different folders.

    Specifically designed to classify sam segmentations from orthogonal branch
    sections into:
     -Lumen
     -Branch
     -Bad
    The bad category is for all other labels that are not the lumen or branch.

    Load the images that you want to classify and the relevant segmentation in a
    napari viewer. The image layer should end with .h5_img and the segmentation layer
    with .h5_sam. The image and segmentation should be in the same folder.

    The Lumen should be labelled as 2, the Branch as 1.

    The save button bad_lumen will crop the image to the bounding box of the lumen,
    while the save button bad will crop the image to the bounding box of the all de-
    tected labels.

    """

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Save Segmentation")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Initialize save paths
        self.lumen_path = ""
        self.branch_path = ""
        self.bad_path = ""

        # Path selectors + display
        self._add_path_selector("Lumen", self.set_lumen_path)
        self._add_path_selector("Branch", self.set_branch_path)
        self._add_path_selector("Bad", self.set_bad_path)

        # Category save buttons
        self._add_save_button("Save as Lumen", self.save_lumen_segmentation)
        self._add_save_button("Save as Branch", self.save_branch_segmentation)
        self._add_save_button("Save as Bad", self.save_bad_segmentation)
        self._add_save_button("Save as Bad Lumen", self.save_bad_lumen)

    def _add_path_selector(self, label, callback):
        btn = QPushButton(f"Set {label} Save Path")
        btn.clicked.connect(callback)
        self.layout.addWidget(btn)

        lbl = QLabel(f"{label} Path: Not set")
        setattr(self, f"{label.lower().replace('/', '_')}_label", lbl)
        self.layout.addWidget(lbl)

    def _add_save_button(self, label, callback):
        """Add a button to save the segmentation."""
        btn = QPushButton(label)
        btn.clicked.connect(callback)
        self.layout.addWidget(btn)

    def set_lumen_path(self):
        """Set the path for lumen save directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Lumen Save Directory")
        if path:
            self.lumen_path = path + "/"
            self.lumen_label.setText(f"Lumen Path: {path}")

    def set_branch_path(self):
        """Set the path for branch save directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Branch Save Directory")
        if path:
            self.branch_path = path + "/"
            self.branch_label.setText(f"Branch Path: {path}")

    def set_bad_path(self):
        """Set the path for bad save directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Bad Save Directory")
        if path:
            self.bad_path = path + "/"
            self.bad_label.setText(f"Bad/Bad Lumen Path: {path}")

    def return_layer(self, path):
        """Return the image and label layer based on the active layer name."""
        current_layer_name = self.viewer.layers.selection.active.name
        name = current_layer_name[:-4]

        if current_layer_name.endswith("img"):
            image_name = current_layer_name
            label_name = name + "_sam"
        elif current_layer_name.endswith("sam"):
            label_name = current_layer_name
            image_name = name + "_img"
        else:
            raise ValueError("Active layer must end with 'img' or 'sam'")

        image = self.viewer.layers[image_name].data
        label = self.viewer.layers[label_name].data
        current_step = self.viewer.dims.current_step[0]

        return (
            image[current_step],
            label[current_step],
            name[:-3] + f"_{current_step}.h5",
        )

    def save_lumen_segmentation(self):
        """Save as lumen segmentation."""
        if not self.lumen_path:
            logger.info("Lumen path not set!")
            return
        logger.info("Saving as lumen segmentation...")
        image, label, name = self.return_layer(self.lumen_path)
        lumen_label = label.copy()
        lumen_label[lumen_label != 2] = 0
        image_masked = image.copy()
        image_masked[lumen_label != 2] = 0

        props = ski.measure.regionprops(ski.measure.label(lumen_label))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            image_masked = image_masked[minr:maxr, minc:maxc]
            lumen_label = lumen_label[minr:maxr, minc:maxc]

        with h5py.File(self.lumen_path + name, "w") as f:
            f.create_dataset("image", data=image_masked)
            f.create_dataset("label", data=lumen_label)

        self._next_slice()

    def save_branch_segmentation(self):
        """Save as branch segmentation."""
        if not self.branch_path:
            logger.info("Branch path not set!")
            return
        logger.info("Saving as branch segmentation...")
        image, label, name = self.return_layer(self.branch_path)
        branch_label = label.copy()
        branch_label[branch_label != 1] = 0
        image_masked = image.copy()
        image_masked[branch_label == 0] = 0

        props = ski.measure.regionprops(ski.measure.label(branch_label))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            image_masked = image_masked[minr:maxr, minc:maxc]
            branch_label = branch_label[minr:maxr, minc:maxc]

        with h5py.File(self.branch_path + name, "w") as f:
            f.create_dataset("image", data=image_masked)
            f.create_dataset("label", data=branch_label)

        self._next_slice()

    def save_bad_segmentation(self):
        """Save as bad segmentation."""
        if not self.bad_path:
            logger.info("Bad path not set!")
            return
        logger.info("Saving as bad segmentation...")
        image, label, name = self.return_layer(self.bad_path)
        bad_label = (label != 0).astype(int)
        image_masked = image * bad_label

        props = ski.measure.regionprops(ski.measure.label(bad_label))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            image_masked = image_masked[minr:maxr, minc:maxc]
            bad_label = bad_label[minr:maxr, minc:maxc]

        with h5py.File(self.bad_path + name, "w") as f:
            f.create_dataset("image", data=image_masked)
            f.create_dataset("label", data=bad_label)

        self._next_slice()

    def save_bad_lumen(self):
        """Save the bad lumen segmentation."""
        if not self.bad_path:
            logger.info("Bad path not set!")
            return
        logger.info("Saving as bad lumen...")
        image, label, name = self.return_layer(self.bad_path)
        bad_label = label.copy()
        bad_label[bad_label != 2] = 0
        image_masked = image.copy()
        image_masked[bad_label != 2] = 0

        props = ski.measure.regionprops(ski.measure.label(bad_label))
        if props:
            minr, minc, maxr, maxc = props[0].bbox
            image_masked = image_masked[minr:maxr, minc:maxc]
            bad_label = bad_label[minr:maxr, minc:maxc]

        with h5py.File(self.bad_path + name, "w") as f:
            f.create_dataset("image", data=image_masked)
            f.create_dataset("label", data=bad_label)

        self._next_slice()

    def _next_slice(self):
        self.viewer.dims.current_step = (
            self.viewer.dims.current_step[0] + 1,
            self.viewer.dims.current_step[1],
            self.viewer.dims.current_step[2],
        )


def add_class_based_on_folder_struct(lumen_path, branch_path, bad_path):
    """
    Add class labels to the h5 files based on their folder structure.

    Parameters
    ----------
    lumen_path : str
        Path to the folder containing lumen files.
    branch_path : str
        Path to the folder containing branch files.
    bad_path : str
        Path to the folder containing bad files.

    """
    # count number of files per class
    class_count = np.array([0, 0, 0], dtype="int32")
    lumen_files = os.listdir(lumen_path)
    for file in lumen_files:
        with h5py.File(lumen_path + file, "a") as f:
            if "class_id" in f:
                del f["class_id"]
            f["class_id"] = np.array([0], dtype="int8")
            class_count[0] += 1

    branch_files = os.listdir(branch_path)
    for file in branch_files:
        with h5py.File(branch_path + file, "a") as f:
            if "class_id" in f:
                del f["class_id"]
            f["class_id"] = np.array([1], dtype="int8")
            class_count[1] += 1
    bad_files = os.listdir(bad_path)
    for file in bad_files:
        with h5py.File(bad_path + file, "a") as f:
            if "class_id" in f:
                del f["class_id"]
            f["class_id"] = np.array([2], dtype="int8")
            class_count[2] += 1
    logger.info("Number of files per class:")
    logger.info(class_count)


def split_and_copy_files(
    file_list, source_dir, train_dir, val_dir, val_split_ratio=0.2
):
    """
    Creates training and validation files from a file list.

    The files are copied to the respective directories.

    Parameters
    ----------
    file_list : list
        List of files to be split.
    source_dir : str
        Directory containing the source files.
    train_dir : str
        Directory to copy training files to.
    val_dir : str
        Directory to copy validation files to.
    val_split_ratio : float
        Ratio of files to be used for validation.
        Default is 0.2 (20% for validation).
    """
    random.shuffle(file_list)
    split_idx = int(len(file_list) * val_split_ratio)
    val_files = file_list[:split_idx]
    train_files = file_list[split_idx:]

    # Copy validation files
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

    # Copy training files
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))


class H5FileDataset(Dataset):
    """
    Dataset reader for HDF5 files.

    Reads images and class labels from HDF5 files.
    Image needs to be stored as "image" and class label as "class_id".
    A bunch of augmentations are applied.


    """

    def __init__(self, file_paths):
        self.model_size = (256, 256)
        self.file_paths = file_paths

        self.transform = Compose(
            [
                Resize(self.model_size),  # Resize to (96, 96)
                ToTensor(),  # Convert to PyTorch tensor (scales [0,255] -> [0,1])
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ResNet normalization
                # Data Augmentations
                RandomApply(
                    [
                        ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.1,
                ),
                RandomApply([RandomRotation(degrees=10)], p=0.1),
                RandomApply([RandomAffine(degrees=0, scale=(0.9, 1.1))], p=0.1),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomResizedCrop(
                    size=self.model_size, scale=(0.8, 1.0), ratio=(0.75, 1.333)
                ),
            ]
        )

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Gets item from the dataset."""
        file_path = self.file_paths[idx]

        # Open the HDF5 file
        with h5py.File(file_path, "r") as h5f:
            img = h5f["image"][:]  # Load image
            class_id = h5f["class_id"][:]  # Scalar value

        img = np.nan_to_num(img, 0)
        # Convert to PIL Image (ensure it's in RGB mode)
        img = grey2rgb(img)
        img = Image.fromarray(img)
        # Apply transformations
        img = self.transform(img)

        # Convert class_id to tensor
        class_id = torch.tensor(class_id, dtype=torch.long).squeeze()

        return img, class_id


class H5DataModule(pl.LightningDataModule):
    """Data module for loading HDF5 files for training and validation."""

    def __init__(self, data_dir="dataset", batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # def setup(self, stage=None):
        # Load all .h5 files from train and val directories
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        train_files = [
            os.path.join(train_dir, f)
            for f in os.listdir(train_dir)
            if f.endswith(".h5")
        ]
        val_files = [
            os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".h5")
        ]

        self.train_dataset = H5FileDataset(train_files)
        self.val_dataset = H5FileDataset(val_files)

    def train_dataloader(self):
        """Returns the training data loader."""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        """Returns the validation data loader."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class ResNet3ClassClassifier(pl.LightningModule):
    """A pl model for classifying images into 3 classes using ResNet50 model."""

    def __init__(self, num_classes=3, pretrained=False):
        super().__init__()
        # Load pre-trained ResNet

        self.resnet = models.resnet50(weights=None)
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.resnet = models.resnet50(weights=weights)

        # Replace the final fully connected layer to output `num_classes` classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # Define accuracy metric
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor after passing through the ResNet model.
        """
        return self.resnet(x)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers for the model.

        This method sets up the Adam optimizer with a learning rate of 1e-4 and
        a StepLR learning rate scheduler that reduces the learning rate by a
        factor of 0.1 every 7 steps.

        Returns
        -------
            tuple: A tuple containing two lists:
                - The first list contains the optimizer(s) to be used.
                - The second list contains the learning rate scheduler(s) to be used.
        """
        # Use Adam optimizer and a learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        images, labels = batch
        logits = self(images)
        # here we can add weights to the loss function to counteract class imbalance
        weights = torch.tensor([1, 1, 1], dtype=torch.float32)
        # weights = weights / class_counts
        weights = weights / torch.sum(weights)
        weights = weights.to(self.device)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        acc = self.train_acc(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step during model evaluation.

        Args:
            batch (tuple): A tuple containing the input data (images) and the
            corresponding labels.
            batch_idx (int): The index of the current batch.

        Returns
        -------
            torch.Tensor: The computed loss for the current validation batch.
        Logs:
            val_loss (float): The cross-entropy loss for the validation batch.
            val_acc (float): The accuracy of the model on the validation batch.
        """
        images, labels = batch
        logits = self(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        acc = self.val_acc(logits, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss


class ResNet3ClassPredictor:
    """
    A class for predicting the class of an image using a pre-trained ResNet50 model.

    The model is loaded from a checkpoint file.
    """

    def __init__(self, model_path, num_classes=3, device=None, model_size=(256, 256)):
        # Automatically detect device if not provided
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load the model
        self.model = ResNet3ClassClassifier(num_classes=num_classes, pretrained=False)
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = ResNet3ClassClassifier.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Define the image transformation
        self.transform = Compose(
            [
                Resize(model_size),  # Resize to (224, 224)
                ToTensor(),  # Convert to PyTorch tensor
                EnsureType(),
                # ResNet normalization
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, img):
        """
        Predict the class of an image.

        Parameters
        ----------
        img : numpy.ndarray
            The input image to classify.

        Returns
        -------
        int
            The predicted class index.
        float
            The confidence of the prediction.
        """
        img = np.nan_to_num(img, 0)
        img = grey2rgb(img)
        # Ensure the input is a PIL Image
        img = Image.fromarray(img)

        # Apply the transformations
        img = self.transform(img)

        # Add batch dimension
        img = img.unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(img)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence
