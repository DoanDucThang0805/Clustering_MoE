"""Plant disease classification dataset loader module.

This module provides a PyTorch Dataset implementation for loading tomato plant
disease images with stratified train/validation/test splitting capabilities. It
supports custom image transformations via albumentations or similar pipelines.

Typical usage example:
    >>> from pathlib import Path
    >>> dataset = LoadDataset(
    ...     root_dir=Path('data/tomato-plantdoc-mod'),
    ...     split='train',
    ...     train_ratio=0.8
    ... )
    >>> image, label = dataset[0]
"""

import os
from typing import List, Tuple, Dict, Literal
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PIL import Image
from sklearn.model_selection import train_test_split


class LoadDataset(Dataset):
    """PyTorch Dataset for plant disease classification images.

    This class loads tomato plant disease images from a directory structure organized
    by disease class, and provides automatic stratified train/validation/test splitting
    to ensure balanced class distribution across all splits. It supports custom image
    transformations and optional context feature extraction for advanced use cases.

    Directory Structure:
        The root directory should contain subdirectories named with "Tomato" prefix:

        root_dir/
        ├── Tomato Early blight leaf/
        ├── Tomato leaf/
        ├── Tomato leaf bacterial spot/
        ├── Tomato leaf late blight/
        ├── Tomato leaf mosaic virus/
        ├── Tomato leaf yellow virus/
        ├── Tomato mold leaf/
        └── Tomato Septoria leaf spot/

    Attributes:
        root_dir (Path): Root directory containing disease class subdirectories.
        split (str): Dataset split identifier - 'train', 'validation', or 'test'.
        train_ratio (float): Proportion of data allocated to training set (0.0-1.0).
        image_paths (List[str]): Absolute file paths to images in the selected split.
        labels (List[int]): Class label indices corresponding to each image in image_paths.
        class_to_idx (Dict[str, int]): Mapping from disease class names to numeric indices.
        idx_to_class (Dict[int, str]): Mapping from numeric indices to disease class names.
        transform (transforms.Compose): Image transformation pipeline (e.g., normalization, augmentation).

    Note:
        - Train/validation/test split ratio: train_ratio / (1-train_ratio)*0.5 / (1-train_ratio)*0.5
        - For example, train_ratio=0.8 results in 80% train, 10% validation, 10% test
        - Stratification ensures each split has balanced class representation
    """

    def __init__(
        self,
        root_dir: Path,
        split: Literal['train', 'validation', 'test'],
        train_ratio: float = 0.8,
        transform: transforms.Compose = None
    ) -> None:
        """Initialize the plant disease dataset loader.

        Loads all images from the root directory, creates class mappings, and splits
        the dataset into train/validation/test sets using stratified sampling to
        maintain balanced class distribution across splits.

        Args:
            root_dir (Path): Root directory containing disease class subdirectories.
                All subdirectories starting with "Tomato" are treated as classes.
            split (str): Dataset split to load - must be 'train', 'validation', or 'test'.
            train_ratio (float, optional): Proportion of data for training set (0.0-1.0).
                Remaining data is split equally between validation and test sets.
                Defaults to 0.8 (80% train, 10% validation, 10% test).
            transform (transforms.Compose, optional): Image transformation pipeline
                (e.g., from albumentations). Applied to each image before returning.
                Expected to follow albumentations API with 'image' key in output dict.
                Defaults to None (no transformations applied).

        Raises:
            ValueError: If split is not one of the valid options ('train', 'validation', 'test').
            FileNotFoundError: If root_dir does not exist or contains no valid image files.

        Example:
            >>> from pathlib import Path
            >>> dataset = LoadDataset(
            ...     root_dir=Path('data/tomato-plantdoc-mod'),
            ...     split='train',
            ...     train_ratio=0.7,
            ...     transform=None
            ... )
            >>> len(dataset)
            1234  # Example number
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.image_paths, self.labels, self.class_to_idx, self.idx_to_class = self._split_dataset()


    def _load_image(self, root_dir: Path) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """Load all images and create class mappings from directory structure.

        Recursively scans the root directory for subdirectories with "Tomato" prefix
        (which represent disease classes), discovers all valid image files within them,
        and creates bidirectional mappings between class names and numeric indices.

        Supported image formats: .png, .jpg, .jpeg (case-insensitive).

        Args:
            root_dir (Path): Root directory containing disease class subdirectories.
                Only directories starting with "Tomato" are processed as valid classes.

        Returns:
            Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
                image_paths (List[str]): Absolute file paths to all discovered images,
                    ordered by class and filename within class.
                labels (List[int]): Class label indices corresponding to each image in
                    image_paths (0-indexed, matching class_to_idx values).
                class_to_idx (Dict[str, int]): Bidirectional mapping from disease class
                    name to numeric index (alphabetically sorted by class name).
                idx_to_class (Dict[int, str]): Inverse mapping from numeric index to
                    disease class name.

        Note:
            - Class names are sorted alphabetically before assigning indices
            - Image files are discovered via listdir, no recursive traversal
            - Invalid image files are silently skipped
        """
        class_names = sorted(
            [d for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d))
             and d.startswith("Tomato")]
        )
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
        image_paths = []
        labels = []
        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, fname))
                    labels.append(class_to_idx[class_name])
        return image_paths, labels, class_to_idx, idx_to_class


    def _split_dataset(self) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
        """Split dataset into stratified train, validation, and test subsets.

        Performs two-stage stratified sampling using sklearn's train_test_split:
        1. First split: train_ratio (default 80%) vs temporary (20%)
        2. Second split: temporary split equally between validation (50%) and test (50%)

        Stratification ensures each class has roughly the same proportion in all splits,
        which is critical for imbalanced datasets. Random state is fixed for reproducibility.

        Split ratio example (with default train_ratio=0.8):
            - Training:   80% of samples
            - Validation: 10% of samples  (50% of remaining 20%)
            - Test:       10% of samples  (50% of remaining 20%)

        Returns:
            Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
                image_paths (List[str]): Image file paths for the selected split
                    (self.split determines which subset is returned).
                labels (List[int]): Numeric class labels corresponding to each image
                    in image_paths.
                class_to_idx (Dict[str, int]): Class name to index mapping (same for
                    all splits, represents all available classes).
                idx_to_class (Dict[int, str]): Index to class name mapping (inverse of
                    class_to_idx).

        Raises:
            ValueError: If self.split is not 'train', 'validation', or 'test'.

        Note:
            - Uses random_state=42 for reproducible splits
            - Stratified sampling on class labels ensures balanced class distribution
        """
        image_paths, labels, class_to_idx, idx_to_class = self._load_image(self.root_dir)

        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size= 1-self.train_ratio, stratify=labels, random_state=42, shuffle=True
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42, shuffle=True
        )

        if self.split == 'train':
            return train_paths, train_labels, class_to_idx, idx_to_class
        elif self.split == 'validation':
            return val_paths, val_labels, class_to_idx, idx_to_class
        elif self.split == 'test':
            return test_paths, test_labels, class_to_idx, idx_to_class
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'")


    def __len__(self) -> int:
        """Get the total number of samples in the current dataset split.

        Returns:
            int: Number of images available in the selected split (train/validation/test).
        """
        return len(self.image_paths)
    
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Retrieve a single sample from the dataset by index.

        Loads an image file from disk and applies optional transformations.

        Image Loading and Processing:
            1. Open image from disk and convert to RGB
            2. Convert to numpy array
            3. Apply transformations if transform pipeline provided
            4. Return transformed image and label

        Args:
            idx (int): Zero-based index into the current split. Must be in range
                [0, len(self)).

        Returns:
            Tuple[np.ndarray, int]:
                image (np.ndarray): Transformed image array.
                    Type and shape depend on the transform pipeline.
                    Without transform: numpy array of shape (H, W, 3) with values in [0, 255]
                    With transform: depends on transform pipeline (typically normalized).
                label (int): Class label index in range [0, num_classes-1].

        Raises:
            IndexError: If idx is out of range [0, len(self)).
            IOError: If image file cannot be read or is corrupted.

        Example:
            >>> dataset = LoadDataset(root_dir=Path('data'), split='train')
            >>> image, label = dataset[0]
            >>> print(image.shape)  # Shape depends on transform
            >>> print(label)  # Class index (e.g., 0-7 for 8 diseases)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_np = augmented["image"]
        return image_np, label
    