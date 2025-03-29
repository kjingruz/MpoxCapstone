"""
Combined dataset loader for skin lesion detection.
Supports both PH2 processed dataset and Mpox dataset with COCO annotations.
"""

import os
import sys
import numpy as np
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import skimage.io
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import tensorflow as tf

# Set up paths for Mask R-CNN (if using the Matterport implementation)
MASK_RCNN_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')
if os.path.exists(MASK_RCNN_DIR):
    sys.path.append(MASK_RCNN_DIR)
    from mrcnn import utils
else:
    # Fallback implementation for dataset without mrcnn
    class DummyUtils:
        class Dataset:
            def __init__(self):
                self.image_info = []
                self.class_info = []
                self._image_ids = []

            def add_class(self, source, class_id, class_name):
                self.class_info.append({"source": source, "id": class_id, "name": class_name})

            def add_image(self, source, image_id, path, **kwargs):
                image_info = {
                    "id": image_id,
                    "source": source,
                    "path": path,
                }
                image_info.update(kwargs)
                self.image_info.append(image_info)
                self._image_ids.append(image_id)

            def prepare(self):
                self.image_ids = self._image_ids

    utils = DummyUtils()

class CombinedLesionDataset(utils.Dataset):
    """
    Dataset class for loading and preparing both PH2 and Mpox datasets for training.
    Handles direct loading from PH2_processed folder structure and COCO annotations for Mpox.
    """

    def load_ph2_processed(self, ph2_dir, subset="train"):
        """
        Load PH2 dataset from processed structure:

        PH2_processed/
        ├── images/              # All images
        ├── masks/               # All masks
        └── split/               # Train/val/test splits
            ├── train/
            │   ├── images/
            │   └── masks/
            ├── val/
            │   ├── images/
            │   └── masks/
            └── test/
                ├── images/
                └── masks/

        Args:
            ph2_dir: Path to PH2_processed directory
            subset: 'train', 'val', or 'test' subset to load
        """
        # Add classes (PH2 contains only skin lesions, no mpox)
        self.add_class("lesion", 1, "skin_lesion")

        # Path to images and masks for the specified subset
        subset_dir = os.path.join(ph2_dir, "split", subset)
        images_dir = os.path.join(subset_dir, "images")
        masks_dir = os.path.join(subset_dir, "masks")

        # Verify directories exist
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"Warning: Could not find {subset} split directories in {ph2_dir}")
            return

        # Load all images from the subset
        print(f"Loading PH2 {subset} images...")
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))])

        for image_file in tqdm(image_files):
            # Image ID (without extension)
            image_id = os.path.splitext(image_file)[0]

            # Image and mask paths
            image_path = os.path.join(images_dir, image_file)

            # Mask filename: Convert IMD002.bmp to IMD002_lesion.bmp
            mask_file = f"{image_id}_lesion{os.path.splitext(image_file)[1]}"
            mask_path = os.path.join(masks_dir, mask_file)

            # Verify mask exists
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {image_file}, skipping")
                continue

            # Add the image
            self.add_image(
                source="lesion",
                image_id=f"ph2_{image_id}",  # Prefix to distinguish from mpox images
                path=image_path,
                mask_path=mask_path,
                dataset="ph2"
            )

        print(f"Loaded {len(image_files)} PH2 {subset} images")

    def load_mpox_coco(self, mpox_dir, subset_ratio=None):
        """
        Load Mpox dataset with COCO annotations.

        Expected structure:
        Monkey Pox/
        ├── M01_01.jpg, M01_02.jpg, ... (Mpox images)
        └── annotation/
            └── instances_default.json (COCO format annotations)

        Args:
            mpox_dir: Path to 'Monkey Pox' directory
            subset_ratio: Optional dict with 'train', 'val', 'test' ratios for splitting
                          If None, all mpox images are loaded
        """
        # Add classes
        if not any(c['name'] == 'mpox' for c in self.class_info):
            self.add_class("mpox", 2, "mpox")

        # Path to images and COCO annotations
        images_dir = mpox_dir
        annotations_file = os.path.join(mpox_dir, "annotation", "instances_default.json")

        # Verify files exist
        if not os.path.exists(annotations_file):
            print(f"Warning: Could not find annotation file at {annotations_file}")
            return

        # Load COCO annotations
        coco = COCO(annotations_file)

        # Get all image IDs
        image_ids = list(coco.imgs.keys())

        # Split images if subset_ratio is provided
        if subset_ratio:
            # Shuffle image IDs for random split
            random.shuffle(image_ids)

            # Calculate split indices
            train_end = int(len(image_ids) * subset_ratio.get('train', 0.7))
            val_end = train_end + int(len(image_ids) * subset_ratio.get('val', 0.15))

            # Get subset image IDs
            if subset_ratio.get('subset') == 'train':
                image_ids = image_ids[:train_end]
            elif subset_ratio.get('subset') == 'val':
                image_ids = image_ids[train_end:val_end]
            elif subset_ratio.get('subset') == 'test':
                image_ids = image_ids[val_end:]

        print(f"Loading Mpox images...")
        loaded_count = 0

        for img_id in tqdm(image_ids):
            # Get image info from COCO
            img_info = coco.loadImgs(img_id)[0]
            image_path = os.path.join(images_dir, img_info['file_name'])

            # Check if image file exists (skip if not found)
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            # Skip images without annotations
            if not anns:
                print(f"Warning: No annotations found for {img_info['file_name']}")
                continue

            # Add the image and its annotations
            self.add_image(
                source="mpox",
                image_id=f"mpox_{img_id}",  # Prefix to distinguish from ph2 images
                path=image_path,
                coco_annotations=anns,
                width=img_info['width'],
                height=img_info['height'],
                dataset="mpox"
            )
            loaded_count += 1

        print(f"Loaded {loaded_count} Mpox images")

    def load_mask(self, image_idx):
        """
        Generate instance masks for an image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Get image info
        info = self.image_info[image_idx]

        # Handle different dataset sources
        if info["dataset"] == "ph2":
            # Load mask from mask_path
            mask = skimage.io.imread(info['mask_path'])

            # PH2 has only one instance per image (binary mask)
            # Convert to boolean array
            mask = mask > 0

            # Add batch dimension for instances (only one instance)
            mask = np.expand_dims(mask, axis=2)

            # All instances are class 1 (skin lesion)
            class_ids = np.array([1], dtype=np.int32)

            return mask, class_ids

        elif info["dataset"] == "mpox":
            # Handle COCO annotations
            # Get image dimensions
            width = info['width']
            height = info['height']

            # Initialize arrays for masks and class IDs
            mask = np.zeros([height, width, len(info['coco_annotations'])], dtype=np.bool)
            class_ids = np.zeros([len(info['coco_annotations'])], dtype=np.int32)

            # Process each annotation
            for i, ann in enumerate(info['coco_annotations']):
                # Convert COCO segmentation to mask
                if isinstance(ann['segmentation'], list):  # Polygon
                    rles = maskUtils.frPyObjects(ann['segmentation'], height, width)
                    m = maskUtils.decode(rles)

                    # If there are multiple polygons, combine them
                    if m.ndim == 3:
                        m = np.sum(m, axis=2) > 0
                else:  # RLE
                    m = maskUtils.decode(ann['segmentation'])

                # Set the mask for this instance
                mask[:, :, i] = m

                # Set class ID (always 2 for mpox)
                class_ids[i] = 2

            return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


class DatasetGenerator:
    """
    Generator class to create TensorFlow dataset from image/mask pairs.
    Use this if you want to use TensorFlow's data pipeline instead of Matterport's Dataset.
    """

    def __init__(self, dataset, batch_size=1, image_size=(512, 512), augment=False):
        """
        Initialize the dataset generator.

        Args:
            dataset: CombinedLesionDataset instance
            batch_size: Number of images per batch
            image_size: Target size for images (height, width)
            augment: Whether to apply data augmentation
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment

    def create_tf_dataset(self):
        """
        Create a TensorFlow dataset that yields batches of images and masks.

        Returns:
            TensorFlow dataset object
        """
        def generator():
            # Loop through all images in the dataset
            for idx in range(len(self.dataset.image_ids)):
                # Get image and mask
                image = self.load_image(idx)
                masks, class_ids = self.dataset.load_mask(idx)

                # Combine all instance masks into one multi-class mask
                # Background = 0, skin lesion = 1, mpox = 2
                combined_mask = np.zeros((*self.image_size, 1), dtype=np.float32)

                for i in range(masks.shape[2]):
                    # Set pixel values based on class
                    mask = np.expand_dims(masks[:, :, i], axis=-1)
                    combined_mask = np.maximum(combined_mask, mask * class_ids[i])

                yield image, combined_mask

        # Create TensorFlow dataset
        output_types = (tf.float32, tf.float32)
        output_shapes = ((self.image_size[0], self.image_size[1], 3),
                        (self.image_size[0], self.image_size[1], 1))

        tf_dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=output_types,
            output_shapes=output_shapes
        )

        # Batch and prefetch
        tf_dataset = tf_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return tf_dataset

    def load_image(self, image_idx):
        """
        Load and preprocess an image from the dataset.

        Args:
            image_idx: Index of the image in the dataset

        Returns:
            Preprocessed image as numpy array
        """
        # Get image path
        image_path = self.dataset.image_info[image_idx]['path']

        # Load and resize image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply augmentation if enabled
        if self.augment:
            image = self._apply_augmentation(image)

        return image

    def _apply_augmentation(self, image):
        """Apply data augmentation to an image"""
        # Example augmentations (can be expanded)
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image)

        # Random brightness adjustment
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)

        return image


def visualize_dataset_samples(dataset, num_samples=5, save_path=None):
    """
    Visualize random samples from the dataset.

    Args:
        dataset: CombinedLesionDataset instance
        num_samples: Number of samples to visualize
        save_path: Optional path to save visualization
    """
    # Prepare figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    # Get random image IDs
    if len(dataset.image_ids) > 0:
        image_indices = random.sample(range(len(dataset.image_ids)), min(num_samples, len(dataset.image_ids)))
    else:
        print("Dataset is empty, nothing to visualize.")
        return

    # Class names lookup
    class_names = {0: "BG", 1: "skin_lesion", 2: "mpox"}

    for i, idx in enumerate(image_indices):
        # Get image info
        info = dataset.image_info[idx]

        # Load image
        image = skimage.io.imread(info['path'])

        # Load mask
        mask, class_ids = dataset.load_mask(idx)

        # Create colored mask with different colors for different classes
        colored_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)

        for j in range(mask.shape[2]):
            if class_ids[j] == 1:  # skin lesion
                colored_mask[mask[:, :, j], :] = [0, 255, 0]  # Green
            elif class_ids[j] == 2:  # mpox
                colored_mask[mask[:, :, j], :] = [255, 0, 0]  # Red

        # Create overlay image
        overlay = image.copy()
        alpha = 0.5
        mask_area = np.any(mask, axis=2)
        overlay[mask_area] = overlay[mask_area] * (1 - alpha) + colored_mask[mask_area] * alpha

        # Plot images
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Original Image: {os.path.basename(info['path'])}")
        axes[i, 0].axis('off')

        # Plot masks (each instance in a different color)
        axes[i, 1].imshow(colored_mask)
        axes[i, 1].set_title(f"Mask (Classes: {', '.join([class_names[c] for c in class_ids])})")
        axes[i, 1].axis('off')

        # Plot overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

    plt.show()


# Example usage
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process skin lesion datasets')
    parser.add_argument('--ph2_dir', required=True, help='Directory containing PH2_processed dataset')
    parser.add_argument('--mpox_dir', required=True, help='Directory containing Monkey Pox dataset')
    parser.add_argument('--output_dir', help='Directory to save visualizations')
    parser.add_argument('--visualize', action='store_true', help='Visualize samples from datasets')
    parser.add_argument('--subset', choices=['train', 'val', 'test'], default='train',
                        help='Which subset to use for PH2 dataset')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset instance
    dataset = CombinedLesionDataset()

    # Load PH2 dataset
    dataset.load_ph2_processed(args.ph2_dir, subset=args.subset)

    # Load Mpox dataset
    # Option 1: Load all mpox images
    dataset.load_mpox_coco(args.mpox_dir)

    # Option 2: Load specific subset of mpox images
    # subset_ratio = {'subset': args.subset, 'train': 0.7, 'val': 0.15, 'test': 0.15}
    # dataset.load_mpox_coco(args.mpox_dir, subset_ratio=subset_ratio)

    # Prepare dataset for use
    dataset.prepare()

    # Print dataset stats
    print("\nDataset Summary:")
    print(f"Number of images: {len(dataset.image_ids)}")
    print(f"Number of classes: {len(dataset.class_info)}")
    for cls in dataset.class_info:
        print(f"  - Class {cls['id']}: {cls['name']}")

    # Visualize samples if requested
    if args.visualize:
        save_path = os.path.join(args.output_dir, f"dataset_samples_{args.subset}.png") if args.output_dir else None
        visualize_dataset_samples(dataset, num_samples=5, save_path=save_path)

    # Example of creating a TensorFlow dataset
    print("\nCreating TensorFlow dataset...")
    data_generator = DatasetGenerator(dataset, batch_size=4, image_size=(512, 512), augment=True)
    tf_dataset = data_generator.create_tf_dataset()
    print(f"TensorFlow dataset created: {tf_dataset}")
