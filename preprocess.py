import os
import shutil
import random
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
import matplotlib.pyplot as plt

class DatasetPreprocessor:
    def __init__(self, output_dir):
        """
        Initialize the dataset preprocessor

        Args:
            output_dir: Directory where the processed dataset will be saved
        """
        self.output_dir = output_dir
        self.processed_images_dir = os.path.join(output_dir, "images")
        self.processed_masks_dir = os.path.join(output_dir, "masks")

        os.makedirs(self.processed_images_dir, exist_ok=True)
        os.makedirs(self.processed_masks_dir, exist_ok=True)

        # Create split directories
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(output_dir, "split", split, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "split", split, "masks"), exist_ok=True)

    def process_ph2_dataset(self, ph2_dir):
        """
        Process the PH2 dataset

        Args:
            ph2_dir: Directory containing the PH2 dataset
        """
        print("Processing PH2 dataset...")

        # Keep track of processed files
        processed_files = []

        for subject in tqdm(os.listdir(ph2_dir)):
            subject_path = os.path.join(ph2_dir, subject)
            if os.path.isdir(subject_path):
                dermo_folder = os.path.join(subject_path, f"{subject}_Dermoscopic_Image")
                lesion_folder = os.path.join(subject_path, f"{subject}_lesion")

                # Check if both folders exist
                if os.path.exists(dermo_folder) and os.path.exists(lesion_folder):
                    # Look for the image and mask files
                    dermo_files = [f for f in os.listdir(dermo_folder) if f.lower().endswith(('.bmp', '.jpg', '.png'))]
                    lesion_files = [f for f in os.listdir(lesion_folder) if f.lower().endswith(('.bmp', '.jpg', '.png'))]

                    if dermo_files and lesion_files:
                        dermo_file = dermo_files[0]
                        lesion_file = lesion_files[0]

                        # Read the image and mask
                        img_path = os.path.join(dermo_folder, dermo_file)
                        mask_path = os.path.join(lesion_folder, lesion_file)

                        # Convert to standard format (PNG)
                        img = Image.open(img_path).convert('RGB')
                        mask = Image.open(mask_path).convert('L')  # Convert to grayscale

                        # Threshold the mask to ensure binary
                        mask_np = np.array(mask)
                        mask_binary = (mask_np > 0).astype(np.uint8) * 255
                        mask = Image.fromarray(mask_binary)

                        # Save with standardized names
                        img_filename = f"ph2_{subject}.png"
                        mask_filename = f"ph2_{subject}_mask.png"

                        img.save(os.path.join(self.processed_images_dir, img_filename))
                        mask.save(os.path.join(self.processed_masks_dir, mask_filename))

                        processed_files.append((img_filename, mask_filename))

        print(f"Processed {len(processed_files)} files from PH2 dataset")
        return processed_files

    def process_coco_dataset(self, coco_json, images_dir):
        """
        Process a dataset with COCO format annotations

        Args:
            coco_json: Path to the COCO JSON file
            images_dir: Directory containing the images referenced in the COCO JSON
        """
        print("Processing COCO format dataset...")

        # Load COCO annotations
        coco = COCO(coco_json)

        # Get all image IDs
        img_ids = coco.getImgIds()
        processed_files = []

        for img_id in tqdm(img_ids):
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']

            # Load image
            img_path = os.path.join(images_dir, img_filename)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found, skipping")
                continue

            img = Image.open(img_path).convert('RGB')

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            if not anns:
                print(f"Warning: No annotations found for {img_filename}, skipping")
                continue

            # Create a mask for each instance
            h, w = img_info['height'], img_info['width']
            instance_masks = []

            for ann in anns:
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], list):  # polygon
                        rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                        mask = maskUtils.decode(rle)
                        # If multiple polygons, combine them
                        if mask.shape[2] > 1:
                            mask = np.sum(mask, axis=2) > 0
                        else:
                            mask = mask[:, :, 0]
                    else:  # RLE
                        mask = maskUtils.decode(ann['segmentation'])

                    instance_masks.append(mask.astype(np.uint8) * 255)

            # Combine instance masks into a single mask with different pixel values for each instance
            if instance_masks:
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                for i, mask in enumerate(instance_masks):
                    # Use i+1 as the label (0 is background)
                    combined_mask[mask > 0] = i + 1

                # Save with standardized names
                base_filename = os.path.splitext(img_filename)[0]
                new_img_filename = f"coco_{base_filename}.png"
                new_mask_filename = f"coco_{base_filename}_mask.png"

                img.save(os.path.join(self.processed_images_dir, new_img_filename))

                # Save instance mask
                mask_img = Image.fromarray(combined_mask)
                mask_img.save(os.path.join(self.processed_masks_dir, new_mask_filename))

                processed_files.append((new_img_filename, new_mask_filename))

        print(f"Processed {len(processed_files)} files from COCO dataset")
        return processed_files

    def split_dataset(self, processed_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split the dataset into train, validation, and test sets

        Args:
            processed_files: List of (image_filename, mask_filename) tuples
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            test_ratio: Proportion of data to use for testing
        """
        print("Splitting dataset...")

        # Shuffle the files
        random.shuffle(processed_files)

        num_total = len(processed_files)
        num_train = int(num_total * train_ratio)
        num_val = int(num_total * val_ratio)

        train_files = processed_files[:num_train]
        val_files = processed_files[num_train:num_train+num_val]
        test_files = processed_files[num_train+num_val:]

        def copy_files(file_list, split_name):
            for img_file, mask_file in tqdm(file_list):
                shutil.copy(
                    os.path.join(self.processed_images_dir, img_file),
                    os.path.join(self.output_dir, "split", split_name, "images", img_file)
                )
                shutil.copy(
                    os.path.join(self.processed_masks_dir, mask_file),
                    os.path.join(self.output_dir, "split", split_name, "masks", mask_file)
                )

        print(f"Copying {len(train_files)} files to train split...")
        copy_files(train_files, "train")

        print(f"Copying {len(val_files)} files to validation split...")
        copy_files(val_files, "val")

        print(f"Copying {len(test_files)} files to test split...")
        copy_files(test_files, "test")

        # Create a JSON file with the split information
        split_info = {
            "train": [f[0] for f in train_files],
            "val": [f[0] for f in val_files],
            "test": [f[0] for f in test_files]
        }

        with open(os.path.join(self.output_dir, "split_info.json"), 'w') as f:
            json.dump(split_info, f, indent=4)

        print("Dataset splitting complete.")

    def visualize_samples(self, num_samples=5):
        """
        Visualize random samples from the processed dataset

        Args:
            num_samples: Number of samples to visualize
        """
        # Get all processed image files
        image_files = os.listdir(self.processed_images_dir)

        # Select random samples
        if len(image_files) > num_samples:
            samples = random.sample(image_files, num_samples)
        else:
            samples = image_files

        # Create a figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))

        for i, img_filename in enumerate(samples):
            # Load image
            img_path = os.path.join(self.processed_images_dir, img_filename)
            img = np.array(Image.open(img_path))

            # Get corresponding mask
            mask_filename = img_filename.replace('.png', '_mask.png')
            mask_path = os.path.join(self.processed_masks_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))

                # Display image and mask
                axes[i, 0].imshow(img)
                axes[i, 0].set_title(f"Image: {img_filename}")
                axes[i, 0].axis('off')

                # For visualization, use colormap for instance masks
                if mask.max() > 1:  # Instance segmentation
                    axes[i, 1].imshow(mask, cmap='tab20')
                    axes[i, 1].set_title(f"Instance Mask: {mask_filename}")
                else:  # Binary mask
                    axes[i, 1].imshow(mask, cmap='gray')
                    axes[i, 1].set_title(f"Binary Mask: {mask_filename}")

                axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sample_visualization.png"))
        plt.close()
        print(f"Visualization saved to {os.path.join(self.output_dir, 'sample_visualization.png')}")


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess datasets for lesion detection")
    parser.add_argument('--output_dir', required=True, help="Directory to save processed dataset")
    parser.add_argument('--ph2_dir', help="Directory containing PH2 dataset")
    parser.add_argument('--coco_json', help="Path to COCO format annotation JSON file")
    parser.add_argument('--coco_images_dir', help="Directory containing images referenced in COCO JSON")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument('--val_ratio', type=float, default=0.15, help="Proportion of data for validation")
    parser.add_argument('--visualize', action='store_true', help="Visualize sample processed images")

    args = parser.parse_args()

    # Check that at least one dataset is provided
    if not args.ph2_dir and not (args.coco_json and args.coco_images_dir):
        parser.error("Please provide at least one dataset (--ph2_dir or --coco_json and --coco_images_dir)")

    # Initialize the preprocessor
    preprocessor = DatasetPreprocessor(args.output_dir)

    # Process datasets
    all_processed_files = []

    if args.ph2_dir:
        ph2_files = preprocessor.process_ph2_dataset(args.ph2_dir)
        all_processed_files.extend(ph2_files)

    if args.coco_json and args.coco_images_dir:
        coco_files = preprocessor.process_coco_dataset(args.coco_json, args.coco_images_dir)
        all_processed_files.extend(coco_files)

    # Split the dataset
    preprocessor.split_dataset(
        all_processed_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio
    )

    # Visualize samples if requested
    if args.visualize:
        preprocessor.visualize_samples()


if __name__ == "__main__":
    main()
