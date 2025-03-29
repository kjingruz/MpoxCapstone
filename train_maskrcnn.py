"""
Mask R-CNN training script for lesion detection.
This uses the Matterport Mask R-CNN implementation.
"""

import os
import sys
import json
import numpy as np
import skimage.io
import datetime
import argparse
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

# Set up paths for Mask R-CNN
# Clone from: https://github.com/matterport/Mask_RCNN
MASK_RCNN_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')
sys.path.append(MASK_RCNN_DIR)

# Import Mask R-CNN modules
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to pretrained COCO weights
COCO_WEIGHTS_PATH = os.path.join(MASK_RCNN_DIR, "mask_rcnn_coco.h5")

# Define Lesion configuration class
class LesionConfig(Config):
    """Configuration for training on the lesion dataset."""
    NAME = "lesion"

    # We use a GPU with 12GB memory, which can fit 2 images with masks.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (background + lesion)
    NUM_CLASSES = 1 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Use smaller anchors because lesions are generally smaller objects
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Use a smaller backbone for faster training
    BACKBONE = "resnet50"

    # Reduce validation steps to speed up training
    VALIDATION_STEPS = 50

    # ROI settings
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 100

    # Use higher resolution images
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512


class LesionDataset(utils.Dataset):
    """Dataset class for lesion segmentation"""

    def load_lesions(self, dataset_dir, subset):
        """Load a subset of the lesion dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class: lesion
        self.add_class("lesion", 1, "lesion")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, "split", subset)

        # Get image and mask directories
        images_dir = os.path.join(subset_dir, "images")
        masks_dir = os.path.join(subset_dir, "masks")

        # Load images and add them to the dataset
        for filename in os.listdir(images_dir):
            if not filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            image_id = filename.split('.')[0]  # Remove file extension
            image_path = os.path.join(images_dir, filename)

            # Corresponding mask path
            mask_filename = filename.replace('.png', '_mask.png')
            mask_path = os.path.join(masks_dir, mask_filename)

            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {filename}, skipping")
                continue

            # Add the image to the dataset
            self.add_image(
                "lesion",
                image_id=image_id,
                path=image_path,
                mask_path=mask_path
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        # Load mask file
        mask = skimage.io.imread(info['mask_path'])

        # Convert grayscale mask to binary masks for each instance
        # In the mask image, each instance has a unique pixel value
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]  # Remove background (0)

        # Create a binary mask for each instance
        masks = np.zeros([mask.shape[0], mask.shape[1], len(instance_ids)], dtype=np.bool)
        class_ids = np.ones([len(instance_ids)], dtype=np.int32)  # All instances are class 1 (lesion)

        for i, instance_id in enumerate(instance_ids):
            masks[:, :, i] = mask == instance_id

        return masks, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


def train(args):
    """Train the model."""
    # Configurations
    config = LesionConfig()
    if args.batch_size:
        config.IMAGES_PER_GPU = args.batch_size
    if args.steps_per_epoch:
        config.STEPS_PER_EPOCH = args.steps_per_epoch

    config.display()

    # Create model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.model_dir)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file if needed
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print(f"Loading weights from {weights_path}")
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a different number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
        ])
    else:
        model.load_weights(weights_path, by_name=True)

    # Training dataset
    dataset_train = LesionDataset()
    dataset_train.load_lesions(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LesionDataset()
    dataset_val.load_lesions(args.dataset, "val")
    dataset_val.prepare()

    # Image Augmentation
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),      # horizontal flips
        iaa.Flipud(0.3),      # vertical flips
        iaa.OneOf([
            iaa.Affine(rotate=(-20, 20)),
            iaa.Affine(scale=(0.8, 1.2))
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 0.5)),
            iaa.Sharpen(alpha=(0, 0.3), lightness=(0.8, 1.2))
        ]),
        iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.03*255))),
        iaa.Sometimes(0.3, iaa.LinearContrast((0.8, 1.2))),
    ])

    # Train the head branches (not the backbone)
    # This is to warm up the model and initialize the new layers
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',
                augmentation=augmentation)

    # Fine-tune all layers (including backbone)
    print("Fine-tuning all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=args.epochs,
                layers='all',
                augmentation=augmentation)

    # Save final model
    final_model_path = os.path.join(args.model_dir, "mask_rcnn_lesion_final.h5")
    model.keras_model.save_weights(final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN for lesion detection.')
    parser.add_argument('--dataset', required=True,
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        help='Path to weights .h5 file or "coco" or "last" or "imagenet"')
    parser.add_argument('--model_dir', required=True,
                        help='Path to save logs and trained model')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Images per GPU (adjusts IMAGES_PER_GPU in config)')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='Steps per epoch (adjusts STEPS_PER_EPOCH in config)')

    args = parser.parse_args()

    # Train the model
    train(args)
