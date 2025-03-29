"""
Training script for the combined skin lesion detection model.

This script incorporates both PH2 and Mpox datasets into a single training pipeline
using the Matterport Mask R-CNN implementation.
"""

import os
import sys
import datetime
import argparse
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

# Import the dataset loader
from data_loader import CombinedLesionDataset, visualize_dataset_samples

# Set up paths for Mask R-CNN
MASK_RCNN_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')
if not os.path.exists(MASK_RCNN_DIR):
    raise ImportError(f"Mask R-CNN directory not found at {MASK_RCNN_DIR}. Please clone matterport/Mask_RCNN.")

sys.path.append(MASK_RCNN_DIR)

# Import Mask R-CNN modules
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize


class LesionConfig(Config):
    """Configuration for training on the lesion datasets."""
    # Give the configuration a recognizable name
    NAME = "lesion"
    
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust according to your GPU memory
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    # Number of classes (background + skin lesion + mpox)
    NUM_CLASSES = 1 + 2
    
    # Number of training steps per epoch
    # This is typically dataset size / batch size
    STEPS_PER_EPOCH = 200
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # Use smaller anchors because lesions are generally smaller objects
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    
    # Use a smaller backbone for faster training
    BACKBONE = "resnet50"
    
    # Reduce validation steps to speed up training
    VALIDATION_STEPS = 50
    
    # ROI settings for better small object detection
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 100
    
    # Use higher resolution images 
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Learning rate and momentum
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001


def train_model(args):
    """
    Train the Mask R-CNN model using both PH2 and Mpox datasets.
    
    Args:
        args: Command line arguments containing paths and training parameters
    """
    # Check for GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPUs")
        for gpu in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  - {gpu.name}: Memory growth enabled")
            except:
                print(f"  - {gpu.name}: Error configuring memory growth")
    else:
        print("No GPUs found. Training will be slow.")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Create logs directory for Mask R-CNN
    logs_dir = os.path.join(args.output_dir, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
    
    # Create model directory
    model_dir = os.path.join(args.output_dir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    
    # Create config
    config = LesionConfig()
    
    # Update config based on command line arguments
    if args.batch_size:
        config.IMAGES_PER_GPU = args.batch_size
    
    if args.steps_per_epoch:
        config.STEPS_PER_EPOCH = args.steps_per_epoch
    
    # Display config
    config.display()
    
    # Create training dataset
    train_dataset = CombinedLesionDataset()
    train_dataset.load_ph2_processed(args.ph2_dir, subset="train")
    train_dataset.load_mpox_coco(args.mpox_dir, 
                                subset_ratio={'subset': 'train', 'train': 0.7, 'val': 0.15, 'test': 0.15})
    train_dataset.prepare()
    
    # Create validation dataset
    val_dataset = CombinedLesionDataset()
    val_dataset.load_ph2_processed(args.ph2_dir, subset="val")
    val_dataset.load_mpox_coco(args.mpox_dir, 
                              subset_ratio={'subset': 'val', 'train': 0.7, 'val': 0.15, 'test': 0.15})
    val_dataset.prepare()
    
    # Print dataset stats
    print("\nTraining Dataset:")
    print(f"Number of images: {len(train_dataset.image_ids)}")
    print(f"Number of classes: {len(train_dataset.class_info)}")
    for cls in train_dataset.class_info:
        print(f"  - Class {cls['id']}: {cls['name']}")
    
    print("\nValidation Dataset:")
    print(f"Number of images: {len(val_dataset.image_ids)}")
    
    # Visualize a few training samples
    if args.visualize:
        print("\nVisualizing training samples...")
        train_viz_path = os.path.join(args.output_dir, "train_samples.png")
        visualize_dataset_samples(train_dataset, num_samples=5, save_path=train_viz_path)
        
        print("\nVisualizing validation samples...")
        val_viz_path = os.path.join(args.output_dir, "val_samples.png")
        visualize_dataset_samples(val_dataset, num_samples=5, save_path=val_viz_path)
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=logs_dir)
    
    # Which weights to start with (imagenet, coco, last or a specific file)
    if args.weights.lower() == "coco":
        weights_path = os.path.join(MASK_RCNN_DIR, "mask_rcnn_coco.h5")
        # Download weights file if needed
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
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
    
    # Data augmentation
    # This will be applied inside the model's training function
    # We define it here for clarity
    augmentation = iaa.SomeOf((0, 3), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(rotate=(-15, 15)),
        iaa.Affine(scale=(0.8, 1.2)),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 0.5)),
        iaa.LinearContrast((0.8, 1.2)),
    ])
    
    # Train the head branches (not the backbone)
    # This is to warm up the model and initialize the new layers
    if not args.skip_heads:
        print("\nTraining network heads")
        model.train(train_dataset, val_dataset,
                    learning_rate=config.LEARNING_RATE,
                    epochs=args.heads_epochs,
                    layers='heads',
                    augmentation=augmentation)
    
    # Fine-tune all layers
    if not args.skip_all:
        print("\nFine-tuning all layers")
        model.train(train_dataset, val_dataset,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=args.all_epochs,
                    layers='all',
                    augmentation=augmentation)
    
    # Save final model
    final_model_path = os.path.join(model_dir, "mask_rcnn_lesion_final.h5")
    model.keras_model.save_weights(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Export to TensorFlow SavedModel format for inference (optional)
    if args.export_tf_saved_model:
        # This requires some extra work since Mask R-CNN wasn't designed for TF SavedModel
        # A simplified approach might be to save just the backbone for feature extraction
        try:
            tf_saved_model_path = os.path.join(model_dir, "tf_saved_model")
            # This is a placeholder - actual implementation would need to handle the complex
            # Mask R-CNN architecture for TF SavedModel export
            print(f"TensorFlow SavedModel export is not fully implemented yet")
        except Exception as e:
            print(f"Error exporting to TensorFlow SavedModel: {e}")
    
    return final_model_path


def inference_test(model_path, dataset, config, num_samples=5, output_dir=None):
    """
    Run inference on a few random images from the dataset to verify the model works.
    
    Args:
        model_path: Path to the trained model weights
        dataset: Dataset to sample images from
        config: Model configuration (modified for inference)
        num_samples: Number of images to test
        output_dir: Directory to save visualization results
    """
    # Create inference config
    inference_config = config.clone()
    inference_config.BATCH_SIZE = 1
    inference_config.IMAGES_PER_GPU = 1
    inference_config.DETECTION_MIN_CONFIDENCE = 0.7
    
    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=None)
    
    # Load weights
    model.load_weights(model_path, by_name=True)
    
    # Get random image IDs
    if len(dataset.image_ids) > 0:
        image_indices = random.sample(range(len(dataset.image_ids)), min(num_samples, len(dataset.image_ids)))
    else:
        print("Dataset is empty, nothing to test.")
        return
    
    # Class names for visualization
    class_names = ["BG", "skin_lesion", "mpox"]
    
    # Run inference on each sampled image
    for i, idx in enumerate(image_indices):
        # Get image info
        info = dataset.image_info[idx]
        image_id = dataset.image_ids[idx]
        
        # Load image
        image = dataset.load_image(image_id)
        
        # Run detection
        results = model.detect([image], verbose=1)[0]
        
        # Visualize results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title(f"Original: {os.path.basename(info['path'])}")
        ax1.axis('off')
        
        # Predictions
        visualize.display_instances(image, 
                                   results['rois'], 
                                   results['masks'], 
                                   results['class_ids'], 
                                   class_names,
                                   results['scores'],
                                   ax=ax2,
                                   title="Predictions")
        
        # Save results if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"inference_test_{i+1}.png"))
        
        plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on lesion datasets')
    
    # Dataset paths
    parser.add_argument('--ph2_dir', required=True, help='Path to PH2_processed directory')
    parser.add_argument('--mpox_dir', required=True, help='Path to Monkey Pox directory')
    
    # Output directory
    parser.add_argument('--output_dir', required=True, help='Directory to save outputs')
    
    # Model weights
    parser.add_argument('--weights', default='coco', 
                        help='Path to weights .h5 file or "coco", "imagenet", "last"')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Images per GPU (overrides config if provided)')
    parser.add_argument('--steps_per_epoch', type=int, default=None,
                        help='Steps per epoch (overrides config if provided)')
    parser.add_argument('--heads_epochs', type=int, default=20, 
                        help='Number of epochs to train heads')
    parser.add_argument('--all_epochs', type=int, default=40, 
                        help='Number of epochs to train all layers')
    
    # Training options
    parser.add_argument('--skip_heads', action='store_true', 
                        help='Skip training the head layers')
    parser.add_argument('--skip_all', action='store_true', 
                        help='Skip training all layers')
    
    # Visualization and testing
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize dataset samples before training')
    parser.add_argument('--test_inference', action='store_true', 
                        help='Run inference test after training')
    
    # Export options
    parser.add_argument('--export_tf_saved_model', action='store_true', 
                        help='Export to TensorFlow SavedModel format')
    
    args = parser.parse_args()
    
    # Train the model
    model_path = train_model(args)
    
    # Test inference if requested
    if args.test_inference:
        # Create test dataset
        test_dataset = CombinedLesionDataset()
        test_dataset.load_ph2_processed(args.ph2_dir, subset="test")
        test_dataset.load_mpox_coco(args.mpox_dir, 
                                  subset_ratio={'subset': 'test', 'train': 0.7, 'val': 0.15, 'test': 0.15})
        test_dataset.prepare()
        
        # Create a test output directory
        test_output_dir = os.path.join(args.output_dir, "inference_tests")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Run inference test
        inference_test(model_path, test_dataset, LesionConfig(), 
                      num_samples=5, output_dir=test_output_dir)
