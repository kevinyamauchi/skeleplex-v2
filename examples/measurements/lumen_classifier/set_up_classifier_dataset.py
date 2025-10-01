import os  # noqa: D100
import shutil

from skeleplex.measurements.lumen_classifier import (
    add_class_based_on_folder_struct,
    split_and_copy_files,
)

# Set paths
lumen_path = "train_data/lumen"
branch_path = "train_data/branch"
bad_path = "train_data/bad"

lumen_files = os.listdir(lumen_path)
branch_files = os.listdir(branch_path)
bad_files = os.listdir(bad_path)

add_class_based_on_folder_struct(
    lumen_path=lumen_path, branch_path=branch_path, bad_path=bad_path
)

# Now split into training and validation data
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/val", exist_ok=True)

# empty the directories
shutil.rmtree("dataset/train")
shutil.rmtree("dataset/val")

# Copy files for each category
split_and_copy_files(lumen_files, lumen_path, "dataset/train", "dataset/val")
split_and_copy_files(branch_files, branch_path, "dataset/train", "dataset/val")
split_and_copy_files(bad_files, bad_path, "dataset/train", "dataset/val")
