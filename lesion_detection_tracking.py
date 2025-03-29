import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from datetime import datetime
import random
from pycocotools import mask as maskUtils
import tensorflow_addons as tfa

class LesionDataGenerator(tf.keras.utils.Sequence):
    """Generator for Mask R-CNN training data"""

    def __init__(self, dataset_dir, batch_size=1, image_size=(512, 512), augment=False, shuffle=True):
        """
        Initialize the data generator

        Args:
            dataset_dir: Directory containing 'images' and 'masks' subdirectories
            batch_size: Number of images per batch
            image_size: Target size for images (height, width)
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle the data between epochs
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle

        self.images_dir = os.path.join(dataset_dir, 'images')
        self.masks_dir = os.path.join(dataset_dir, 'masks')

        self.image_filenames = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        self.indexes = np.arange(len(self.image_filenames))
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_filenames = [self.image_filenames[k] for k in indexes]

        # Generate data
        X, y = self._generate_data(batch_filenames)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_data(self, batch_filenames):
        """Generate a batch of data"""
        X = np.empty((self.batch_size, *self.image_size, 3), dtype=np.float32)

        # Initialize arrays for RCNN targets
        batch_images = []
        batch_image_metas = []
        batch_rpn_match = []
        batch_rpn_bbox = []
        batch_gt_class_ids = []
        batch_gt_boxes = []
        batch_gt_masks = []

        for i, filename in enumerate(batch_filenames):
            # Load image
            img_path = os.path.join(self.images_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.image_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]

            # Find corresponding mask
            mask_filename = filename.replace('.png', '_mask.png')
            mask_path = os.path.join(self.masks_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = mask.resize(self.image_size)
                mask_array = np.array(mask)

                # Apply data augmentation if needed
                if self.augment:
                    img_array, mask_array = self._apply_augmentation(img_array, mask_array)

                # Extract instance masks and bounding boxes
                instance_ids = np.unique(mask_array)
                instance_ids = instance_ids[instance_ids > 0]  # Remove background (0)

                # Process each instance mask
                boxes = []
                masks = []
                class_ids = []

                for instance_id in instance_ids:
                    instance_mask = (mask_array == instance_id).astype(np.uint8)

                    # Find contours to get bounding box
                    contours, _ = cv2.findContours(
                        instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Get largest contour in case of multiple
                        contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(contour)

                        # Skip very small instances
                        if w < 10 or h < 10:
                            continue

                        # Convert to normalized coordinates (y1, x1, y2, x2)
                        box = np.array([y, x, y+h, x+w], dtype=np.float32)
                        box = box / np.array([*self.image_size, *self.image_size])

                        # Add to lists
                        boxes.append(box)
                        masks.append(instance_mask)
                        class_ids.append(1)  # Always class 1 (lesion)

                # Store the image
                X[i] = img_array

                # Convert lists to arrays for batch
                if boxes:
                    boxes = np.array(boxes, dtype=np.float32)
                    masks = np.array(masks, dtype=np.uint8)
                    masks = np.transpose(masks, (1, 2, 0))  # H, W, N
                    class_ids = np.array(class_ids, dtype=np.int32)
                else:
                    # No instances in this image
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    masks = np.zeros((*self.image_size, 0), dtype=np.uint8)
                    class_ids = np.zeros((0), dtype=np.int32)

                # Create RCNN batch components (simplified for demonstration)
                batch_images.append(img_array)
                # Simplified image meta - just shape info
                batch_image_metas.append(np.array([*self.image_size, 3]))
                # Simplified RPN match and bbox - would normally come from RPN
                batch_rpn_match.append(np.ones((256,), dtype=np.int32))
                batch_rpn_bbox.append(np.zeros((256, 4), dtype=np.float32))
                batch_gt_class_ids.append(class_ids)
                batch_gt_boxes.append(boxes)
                batch_gt_masks.append(masks)

        # Return the batch
        # In a real implementation, these would be properly padded and batched
        # This is simplified for demonstration
        return X, {
            "rpn_class_logits": np.zeros((self.batch_size, 1)),  # Dummy output
            "rpn_bbox": np.zeros((self.batch_size, 1)),  # Dummy output
            "mrcnn_class_logits": np.zeros((self.batch_size, 1)),  # Dummy output
            "mrcnn_bbox": np.zeros((self.batch_size, 1)),  # Dummy output
            "mrcnn_mask": np.zeros((self.batch_size, 1))  # Dummy output
        }

    def _apply_augmentation(self, image, mask):
        """Apply data augmentation to image and mask"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # Random vertical flip
        if random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # Random rotation
        angle = random.uniform(-30, 30)
        image = self._rotate_image(image, angle)
        mask = self._rotate_image(mask, angle, is_mask=True)

        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 1)

        return image, mask

    def _rotate_image(self, image, angle, is_mask=False):
        """Rotate image or mask by angle degrees"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        if is_mask:
            rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_NEAREST)
        else:
            rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)

        return rotated


class MaskRCNNLesionDetector:
    """Lesion detection using Mask R-CNN architecture"""

    def __init__(self, config=None):
        """
        Initialize the Mask R-CNN detector

        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = {
            "IMAGE_SIZE": (512, 512),
            "BACKBONE": "resnet50",
            "BATCH_SIZE": 1,
            "LEARNING_RATE": 1e-4,
            "EPOCHS": 50,
            "NUM_CLASSES": 2,  # Background + lesion
            "DETECTION_MIN_CONFIDENCE": 0.9,
            "MAX_INSTANCES": 100,  # Maximum number of instances to detect
            "RPN_ANCHOR_SCALES": (32, 64, 128, 256, 512),
            "TRAIN_ROIS_PER_IMAGE": 200,
            "ROI_POSITIVE_RATIO": 0.33
        }

        # Update with provided config
        if config:
            self.config.update(config)

        # Initialize the model
        self.model = None

    def build_model(self):
        """
        Build the Mask R-CNN model

        Note: This is a simplified implementation. A full Mask R-CNN would use
        a dedicated library like matterport/Mask_RCNN or TensorFlow Object Detection API.
        """
        K.clear_session()

        # Input layer
        input_image = layers.Input(
            shape=[*self.config["IMAGE_SIZE"], 3], name="input_image"
        )

        # BackBone - Feature Extractor (ResNet50)
        backbone = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=input_image
        )

        # Extract feature maps at different levels
        C2 = backbone.get_layer("conv3_block4_out").output
        C3 = backbone.get_layer("conv4_block6_out").output
        C4 = backbone.get_layer("conv5_block3_out").output

        # Create Feature Pyramid Network (FPN)
        P2 = self._create_fpn_level(C2, 256)
        P3 = self._create_fpn_level(C3, 256)
        P4 = self._create_fpn_level(C4, 256)

        # RPN (Region Proposal Network)
        rpn_class_logits, rpn_probs, rpn_bbox = self._build_rpn(P2, P3, P4)

        # Generate proposals
        # For simplicity, in a real implementation this would compute proposal ROIs
        # Here we'll just provide a placeholder
        rois = layers.Input(shape=[None, 4], name="input_rois")

        # ROI Align - extract features for each ROI
        roi_align = self._build_roi_align(P2, P3, P4, rois)

        # Classifier head
        mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = self._build_classifier_head(roi_align)

        # Mask head
        mrcnn_mask = self._build_mask_head(roi_align)

        # Define the full model
        model = models.Model(
            inputs=[input_image, rois],
            outputs=[
                rpn_class_logits, rpn_probs, rpn_bbox,
                mrcnn_class_logits, mrcnn_probs, mrcnn_bbox,
                mrcnn_mask
            ],
            name="mask_rcnn"
        )

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.config["LEARNING_RATE"]),
            loss={
                'rpn_class_logits': self._rpn_class_loss,
                'rpn_bbox': self._rpn_bbox_loss,
                'mrcnn_class_logits': self._mrcnn_class_loss,
                'mrcnn_bbox': self._mrcnn_bbox_loss,
                'mrcnn_mask': self._mrcnn_mask_loss
            },
            loss_weights={
                'rpn_class_logits': 1.0,
                'rpn_bbox': 1.0,
                'mrcnn_class_logits': 1.0,
                'mrcnn_bbox': 1.0,
                'mrcnn_mask': 1.0
            }
        )

        self.model = model
        return model

    def _create_fpn_level(self, input_tensor, filters):
        """Create a Feature Pyramid Network level"""
        x = layers.Conv2D(filters, (1, 1), padding="same")(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def _build_rpn(self, P2, P3, P4):
        """
        Build Region Proposal Network

        Note: This is simplified. A real implementation would have shared
        convolutional layers and multiple anchors.
        """
        # Simplified - in reality this would process each level of FPN
        rpn_feature = layers.concatenate([
            layers.UpSampling2D()(P4),
            P3,
            layers.MaxPooling2D()(P2)
        ])

        # RPN classification (foreground vs background)
        x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(rpn_feature)
        rpn_class_logits = layers.Conv2D(2, (1, 1))(x)
        rpn_probs = layers.Activation("softmax")(rpn_class_logits)

        # RPN bounding box regression
        x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(rpn_feature)
        rpn_bbox = layers.Conv2D(4, (1, 1))(x)

        return rpn_class_logits, rpn_probs, rpn_bbox

    def _build_roi_align(self, P2, P3, P4, rois):
        """
        Build ROI Align operation

        Note: This is a placeholder. Real implementation would use a proper ROI Align layer
        that extracts features for each ROI from the appropriate pyramid level.
        """
        # Simplified - in a real implementation, this would select features from
        # the appropriate pyramid level based on ROI size
        return layers.Lambda(lambda x: x)(rois)  # Placeholder

    def _build_classifier_head(self, roi_align):
        """Build the classifier head"""
        # Simplified - in a real implementation, this would process the roi_align output
        # Here we just use placeholders
        class_logits = layers.Dense(self.config["NUM_CLASSES"])(layers.Flatten()(roi_align))
        probs = layers.Activation("softmax")(class_logits)
        bbox = layers.Dense(4 * (self.config["NUM_CLASSES"] - 1))(layers.Flatten()(roi_align))

        return class_logits, probs, bbox

    def _build_mask_head(self, roi_align):
        """Build the mask head"""
        # Simplified - in a real implementation, this would be a series of conv layers
        # followed by a deconvolution to generate masks
        # Here we just use a placeholder
        mask = layers.Conv2D(self.config["NUM_CLASSES"] - 1, (1, 1))(roi_align)

        return mask

    # Loss functions - simplified placeholders
    def _rpn_class_loss(self, y_true, y_pred):
        return K.mean(y_pred * 0)  # Dummy loss

    def _rpn_bbox_loss(self, y_true, y_pred):
        return K.mean(y_pred * 0)  # Dummy loss

    def _mrcnn_class_loss(self, y_true, y_pred):
        return K.mean(y_pred * 0)  # Dummy loss

    def _mrcnn_bbox_loss(self, y_true, y_pred):
        return K.mean(y_pred * 0)  # Dummy loss

    def _mrcnn_mask_loss(self, y_true, y_pred):
        return K.mean(y_pred * 0)  # Dummy loss

    def train(self, train_dir, val_dir, model_path, epochs=None):
        """
        Train the model

        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            model_path: Path to save the trained model
            epochs: Number of epochs to train (overrides config)
        """
        if epochs:
            self.config["EPOCHS"] = epochs

        # Create data generators
        train_gen = LesionDataGenerator(
            train_dir,
            batch_size=self.config["BATCH_SIZE"],
            image_size=self.config["IMAGE_SIZE"],
            augment=True
        )

        val_gen = LesionDataGenerator(
            val_dir,
            batch_size=self.config["BATCH_SIZE"],
            image_size=self.config["IMAGE_SIZE"],
            augment=False
        )

        # Build the model if it doesn't exist
        if self.model is None:
            self.build_model()

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                verbose=1,
                save_best_only=True
            ),
            EarlyStopping(
                patience=10,
                monitor='val_loss',
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                verbose=1,
                min_lr=1e-6
            )
        ]

        # Train the model
        history = self.model.fit(
            train_gen,
            epochs=self.config["EPOCHS"],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def load_model(self, model_path):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                '_rpn_class_loss': self._rpn_class_loss,
                '_rpn_bbox_loss': self._rpn_bbox_loss,
                '_mrcnn_class_loss': self._mrcnn_class_loss,
                '_mrcnn_bbox_loss': self._mrcnn_bbox_loss,
                '_mrcnn_mask_loss': self._mrcnn_mask_loss
            }
        )
        return self.model


# This is a simplified version of Mask R-CNN. In practice, you would use a full implementation
# like Matterport's Mask R-CNN or TensorFlow Object Detection API.
# The implementation above is meant to illustrate the key components, but lacks many details.

# For actual implementation, I recommend using a tested framework:
def get_real_maskrcnn_implementation():
    """
    Instructions for using a real Mask R-CNN implementation

    This is not functional code, but guidance on how to use existing implementations.
    """
    print("""
    IMPLEMENTING MASK R-CNN FOR LESION DETECTION

    For a working implementation, choose one of these approaches:

    OPTION 1: Use Matterport's Mask R-CNN (Recommended)
    -----------------------------------------------
    1. Install the library:
       pip install git+https://github.com/matterport/Mask_RCNN.git

    2. Import and use:
       ```
       import mrcnn
       from mrcnn import utils
       from mrcnn import model as modellib
       from mrcnn import visualize
       from mrcnn.config import Config

       # Create a configuration class
       class LesionConfig(Config):
           NAME = "lesion"
           IMAGES_PER_GPU = 1
           GPU_COUNT = 1
           NUM_CLASSES = 2  # Background + lesion
           STEPS_PER_EPOCH = 100
           DETECTION_MIN_CONFIDENCE = 0.9
           BACKBONE = "resnet50"

       # Create a dataset class
       class LesionDataset(utils.Dataset):
           def load_lesions(self, dataset_dir, subset):
               # Add classes
               self.add_class("lesion", 1, "lesion")

               # Load annotations from processed dataset
               annotations_file = os.path.join(dataset_dir, f"{subset}_annotations.json")
               with open(annotations_file) as f:
                   annotations = json.load(f)

               # Add images and annotations
               for a in annotations:
                   # Add the image
                   image_path = os.path.join(dataset_dir, a['image_path'])
                   image = skimage.io.imread(image_path)
                   height, width = image.shape[:2]

                   self.add_image(
                       "lesion",
                       image_id=a['image_id'],
                       path=image_path,
                       width=width,
                       height=height,
                       annotations=a['regions']
                   )

           def load_mask(self, image_id):
               # Get image info
               info = self.image_info[image_id]

               # Convert annotations to masks
               count = len(info['annotations'])
               mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
               class_ids = np.zeros(count, dtype=np.int32)

               for i, region in enumerate(info['annotations']):
                   # Get region coordinates
                   y1, x1, y2, x2 = region['bbox']

                   # Create binary mask
                   m = np.zeros([info['height'], info['width']], dtype=np.uint8)
                   m[y1:y2, x1:x2] = region['mask']
                   mask[:, :, i] = m

                   # All instances are of class "lesion"
                   class_ids[i] = 1

               return mask.astype(np.bool), class_ids

       # Create the model
       config = LesionConfig()
       model = modellib.MaskRCNN(mode="training", config=config, model_dir="./logs")

       # Load weights
       model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
           "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
       ])

       # Prepare datasets
       train_dataset = LesionDataset()
       train_dataset.load_lesions("./data", "train")
       train_dataset.prepare()

       val_dataset = LesionDataset()
       val_dataset.load_lesions("./data", "val")
       val_dataset.prepare()

       # Train
       model.train(
           train_dataset,
           val_dataset,
           learning_rate=config.LEARNING_RATE,
           epochs=50,
           layers="heads"
       )
       ```

    OPTION 2: Use TensorFlow Object Detection API
    --------------------------------------------
    1. Install TensorFlow Object Detection API

    2. Use a Mask R-CNN pre-trained model from the model zoo

    3. Configure a pipeline.config file for transfer learning

    4. Convert your dataset to TFRecord format

    5. Train using:
       ```
       !python /path/to/models/research/object_detection/model_main_tf2.py \\
           --pipeline_config_path=/path/to/pipeline.config \\
           --model_dir=/path/to/model_dir \\
           --num_train_steps=50000
       ```

    OPTION 3: Use Detectron2
    ----------------------
    1. Install Detectron2 (PyTorch-based)

    2. Register your dataset

    3. Configure training parameters

    4. Train a Mask R-CNN model

    These frameworks provide complete, tested implementations with all the
    complex parts (ROI Align, loss functions, NMS, etc.) properly implemented.
    """)

    return None


class LesionTracker:
    """
    Track lesions across multiple images for the same patient over time
    """

    def __init__(self, detector, tracking_dir):
        """
        Initialize the lesion tracker

        Args:
            detector: Instance of a lesion detector (e.g., MaskRCNNLesionDetector)
            tracking_dir: Directory to save tracking data
        """
        self.detector = detector
        self.tracking_dir = tracking_dir
        self.tracking_data = {}

        # Create tracking directory if it doesn't exist
        os.makedirs(tracking_dir, exist_ok=True)

    def detect_and_track_lesions(self, image_path, patient_id, timestamp=None):
        """
        Detect lesions in an image and track them for a patient

        Args:
            image_path: Path to the image file
            patient_id: Unique identifier for the patient
            timestamp: Timestamp for the image (if None, current time will be used)

        Returns:
            Dictionary with lesion count, lesion metadata, total area, etc.
        """
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create patient directory if it doesn't exist
        patient_dir = os.path.join(self.tracking_dir, f"patient_{patient_id}")
        os.makedirs(patient_dir, exist_ok=True)

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)

        # Detect lesions (implementation will vary based on detector used)
        # This is a placeholder - actual implementation depends on detector
        lesions = self._detect_lesions(img_array)

        # Calculate total lesion area
        total_area = sum(lesion['area'] for lesion in lesions)

        # Create visualization
        visualization = self._visualize_lesions(img_array, lesions)

        # Prepare result
        result = {
            'patient_id': patient_id,
            'timestamp': timestamp,
            'lesion_count': len(lesions),
            'lesions': lesions,
            'total_area': total_area,
            'visualization': visualization
        }

        # Save tracking information
        self._save_tracking_info(patient_id, timestamp, result, image_path)

        # Update in-memory tracking data
        if patient_id not in self.tracking_data:
            self.tracking_data[patient_id] = []

        self.tracking_data[patient_id].append({
            'timestamp': timestamp,
            'lesion_count': len(lesions),
            'total_area': total_area,
            'lesions': lesions
        })

        return result

    def _detect_lesions(self, image):
        """
        Detect lesions in an image

        Args:
            image: Image as numpy array

        Returns:
            List of lesion dictionaries with metadata
        """
        # This is a placeholder - implementation depends on detector
        # For a real Mask R-CNN, you would:
        # 1. Preprocess the image
        # 2. Run detection
        # 3. Process the outputs to extract lesion information

        # Placeholder implementation
        return [
            {
                'id': 1,
                'score': 0.95,
                'bbox': [100, 100, 150, 150],  # [x1, y1, x2, y2]
                'area': 2500,
                'centroid': [125, 125]
            }
        ]

    def _visualize_lesions(self, image, lesions):
        """
        Create a visualization of detected lesions

        Args:
            image: Image as numpy array
            lesions: List of lesion dictionaries

        Returns:
            Image with lesions visualized
        """
        # Create a copy of the image
        vis_img = image.copy()

        # Draw each lesion
        for i, lesion in enumerate(lesions):
            # Get bounding box
            x1, y1, x2, y2 = lesion['bbox']

            # Draw rectangle
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add lesion ID and confidence
            text = f"#{i+1} ({lesion['score']:.2f})"
            cv2.putText(vis_img, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add count information
        text = f"Lesion count: {len(lesions)}"
        cv2.putText(vis_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        return vis_img

    def _save_tracking_info(self, patient_id, timestamp, result, original_image_path):
        """
        Save tracking information for a patient

        Args:
            patient_id: Unique identifier for the patient
            timestamp: Timestamp for the image
            result: Detection result dictionary
            original_image_path: Path to the original image
        """
        # Create patient directory if it doesn't exist
        patient_dir = os.path.join(self.tracking_dir, f"patient_{patient_id}")
        os.makedirs(patient_dir, exist_ok=True)

        # Create a copy of the result (without the visualization)
        result_copy = result.copy()

        # Remove the visualization from the JSON (save separately)
        visualization = result_copy.pop('visualization')

        # Save visualization as an image
        vis_path = os.path.join(patient_dir, f"{timestamp}_visualization.png")
        plt.imsave(vis_path, visualization)

        # Copy original image
        orig_filename = os.path.basename(original_image_path)
        orig_dest_path = os.path.join(patient_dir, f"{timestamp}_{orig_filename}")
        shutil.copy(original_image_path, orig_dest_path)

        # Add paths to the JSON
        result_copy['visualization_path'] = vis_path
        result_copy['original_image_path'] = orig_dest_path

        # Save the JSON
        with open(os.path.join(patient_dir, f"{timestamp}_data.json"), 'w') as f:
            json.dump(result_copy, f, indent=4)

    def get_patient_history(self, patient_id):
        """
        Get the tracking history for a patient

        Args:
            patient_id: Unique identifier for the patient

        Returns:
            List of tracking data dictionaries, ordered by timestamp
        """
        patient_dir = os.path.join(self.tracking_dir, f"patient_{patient_id}")

        if not os.path.exists(patient_dir):
            return None

        # Load all JSON files for the patient
        data_files = [f for f in os.listdir(patient_dir) if f.endswith('_data.json')]
        data_files.sort()  # Sort by timestamp

        history = []
        for file in data_files:
            with open(os.path.join(patient_dir, file), 'r') as f:
                data = json.load(f)
                history.append(data)

        return history

    def analyze_progression(self, patient_id):
        """
        Analyze the progression of lesions over time for a patient

        Args:
            patient_id: Unique identifier for the patient

        Returns:
            Dictionary with progression analysis
        """
        history = self.get_patient_history(patient_id)

        if not history or len(history) < 2:
            return {"status": "Not enough data for progression analysis"}

        # Get the first and last records
        first_record = history[0]
        last_record = history[-1]

        # Calculate changes
        count_change = last_record['lesion_count'] - first_record['lesion_count']
        area_change = last_record['total_area'] - first_record['total_area']

        # Calculate percentage changes
        if first_record['lesion_count'] > 0:
            count_change_pct = (count_change / first_record['lesion_count']) * 100
        else:
            count_change_pct = float('inf') if count_change > 0 else 0

        if first_record['total_area'] > 0:
            area_change_pct = (area_change / first_record['total_area']) * 100
        else:
            area_change_pct = float('inf') if area_change > 0 else 0

        # Determine progression status
        if count_change < 0 and area_change < 0:
            status = "Improving"
        elif count_change > 0 and area_change > 0:
            status = "Worsening"
        else:
            status = "Mixed"

        # Prepare time series data for visualization
        time_points = [record['timestamp'] for record in history]
        lesion_counts = [record['lesion_count'] for record in history]
        total_areas = [record['total_area'] for record in history]

        # Create visualization of time series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot lesion count
        ax1.plot(range(len(time_points)), lesion_counts, 'b-o')
        ax1.set_xticks(range(len(time_points)))
        ax1.set_xticklabels([t.split('_')[0] for t in time_points], rotation=45)
        ax1.set_ylabel('Lesion Count')
        ax1.set_title('Lesion Count Over Time')
        ax1.grid(True)

        # Plot total area
        ax2.plot(range(len(time_points)), total_areas, 'r-o')
        ax2.set_xticks(range(len(time_points)))
        ax2.set_xticklabels([t.split('_')[0] for t in time_points], rotation=45)
        ax2.set_ylabel('Total Lesion Area (pixels)')
        ax2.set_title('Total Lesion Area Over Time')
        ax2.grid(True)

        plt.tight_layout()

        # Save the chart
        chart_path = os.path.join(self.tracking_dir, f"patient_{patient_id}", "progression_chart.png")
        plt.savefig(chart_path)
        plt.close()

        return {
            "patient_id": patient_id,
            "time_period": f"{first_record['timestamp']} to {last_record['timestamp']}",
            "initial_count": first_record['lesion_count'],
            "final_count": last_record['lesion_count'],
            "count_change": count_change,
            "count_change_pct": count_change_pct,
            "initial_area": first_record['total_area'],
            "final_area": last_record['total_area'],
            "area_change": area_change,
            "area_change_pct": area_change_pct,
            "status": status,
            "chart_path": chart_path
        }

    def match_lesions_between_images(self, patient_id, timestamp1, timestamp2):
        """
        Match lesions between two images to track individual lesions

        Args:
            patient_id: Unique identifier for the patient
            timestamp1: Timestamp for the first image
            timestamp2: Timestamp for the second image

        Returns:
            Dictionary with lesion matching information
        """
        patient_dir = os.path.join(self.tracking_dir, f"patient_{patient_id}")

        # Load data for both timestamps
        try:
            with open(os.path.join(patient_dir, f"{timestamp1}_data.json"), 'r') as f:
                data1 = json.load(f)

            with open(os.path.join(patient_dir, f"{timestamp2}_data.json"), 'r') as f:
                data2 = json.load(f)
        except FileNotFoundError:
            return {"status": "Timestamp data not found"}

        # Match lesions based on centroid proximity
        matches = []
        for lesion1 in data1['lesions']:
            best_match = None
            min_distance = float('inf')

            for lesion2 in data2['lesions']:
                # Calculate distance between centroids
                c1 = np.array(lesion1['centroid'])
                c2 = np.array(lesion2['centroid'])
                distance = np.linalg.norm(c1 - c2)

                # Consider it a match if the distance is below a threshold
                # and it's the closest match so far
                if distance < 50 and distance < min_distance:  # Threshold of 50 pixels
                    min_distance = distance
                    best_match = lesion2

            if best_match:
                # Calculate area change
                area_change = best_match['area'] - lesion1['area']
                area_change_pct = (area_change / lesion1['area']) * 100 if lesion1['area'] > 0 else 0

                matches.append({
                    "lesion_id_1": lesion1['id'],
                    "lesion_id_2": best_match['id'],
                    "area_1": lesion1['area'],
                    "area_2": best_match['area'],
                    "area_change": area_change,
                    "area_change_pct": area_change_pct,
                    "distance": min_distance
                })

        # Find new and disappeared lesions
        matched_ids_2 = [m['lesion_id_2'] for m in matches]
        new_lesions = [l for l in data2['lesions'] if l['id'] not in matched_ids_2]

        matched_ids_1 = [m['lesion_id_1'] for m in matches]
        disappeared_lesions = [l for l in data1['lesions'] if l['id'] not in matched_ids_1]

        # Generate a visualization comparing the two images
        img1 = plt.imread(data1['original_image_path'])
        img2 = plt.imread(data2['original_image_path'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Display first image with lesions
        ax1.imshow(img1)
        ax1.set_title(f"First image ({timestamp1})")

        # Display second image with lesions
        ax2.imshow(img2)
        ax2.set_title(f"Second image ({timestamp2})")

        # Add annotations for matched lesions
        for match in matches:
            # Get lesion info
            lesion1 = next((l for l in data1['lesions'] if l['id'] == match['lesion_id_1']), None)
            lesion2 = next((l for l in data2['lesions'] if l['id'] == match['lesion_id_2']), None)

            if lesion1 and lesion2:
                # Draw circles and add labels
                x1, y1 = lesion1['centroid']
                x2, y2 = lesion2['centroid']

                ax1.add_patch(plt.Circle((x1, y1), 10, color='green', fill=False, linewidth=2))
                ax1.text(x1, y1, f"#{match['lesion_id_1']}", color='white',
                         backgroundcolor='green', fontsize=8)

                ax2.add_patch(plt.Circle((x2, y2), 10, color='green', fill=False, linewidth=2))
                ax2.text(x2, y2, f"#{match['lesion_id_2']}", color='white',
                         backgroundcolor='green', fontsize=8)

        # Annotate disappeared lesions
        for lesion in disappeared_lesions:
            x, y = lesion['centroid']
            ax1.add_patch(plt.Circle((x, y), 10, color='red', fill=False, linewidth=2))
            ax1.text(x, y, f"#{lesion['id']}", color='white',
                     backgroundcolor='red', fontsize=8)

        # Annotate new lesions
        for lesion in new_lesions:
            x, y = lesion['centroid']
            ax2.add_patch(plt.Circle((x, y), 10, color='blue', fill=False, linewidth=2))
            ax2.text(x, y, f"#{lesion['id']}", color='white',
                     backgroundcolor='blue', fontsize=8)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='green', markersize=10, label='Matched'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='red', markersize=10, label='Disappeared'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='blue', markersize=10, label='New')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        # Save the comparison
        comparison_path = os.path.join(
            patient_dir, f"comparison_{timestamp1}_{timestamp2}.png"
        )
        plt.savefig(comparison_path)
        plt.close()

        return {
            "patient_id": patient_id,
            "timestamp1": timestamp1,
            "timestamp2": timestamp2,
            "matched_lesions": matches,
            "new_lesions_count": len(new_lesions),
            "disappeared_lesions_count": len(disappeared_lesions),
            "comparison_path": comparison_path
        }


# Example usage
def example_pipeline():
    """
    Example of the full lesion detection and tracking pipeline
    """
    # 1. Preprocess the datasets
    print("Step 1: Preprocess the datasets")
    from dataset_preprocessing import DatasetPreprocessor

    # Initialize preprocessor
    preprocessor = DatasetPreprocessor("./processed_dataset")

    # Process PH2 dataset
    ph2_files = preprocessor.process_ph2_dataset("./PH2_Dataset_images")

    # Process COCO format dataset
    coco_files = preprocessor.process_coco_dataset(
        "./annotations.json",
        "./images"
    )

    # Combine and split the datasets
    all_files = ph2_files + coco_files
    preprocessor.split_dataset(all_files)

    # 2. Train the Mask R-CNN model
    print("\nStep 2: Train the Mask R-CNN model")
    # In practice, use a proper Mask R-CNN implementation like Matterport's
    get_real_maskrcnn_implementation()

    # 3. Set up the lesion tracker
    print("\nStep 3: Set up the lesion tracker")
    tracker = LesionTracker(None, "./tracking_data")

    # 4. Track lesions for a patient over time
    print("\nStep 4: Track lesions for a patient over time")
    # This would be called for each new image from the patient
    # tracker.detect_and_track_lesions("./sample_image.jpg", "patient_001")

    # 5. Analyze progression
    print("\nStep 5: Analyze progression")
    # After collecting multiple timepoints, analyze progression
    # progression = tracker.analyze_progression("patient_001")
    # print(f"Progression status: {progression['status']}")

    print("\nExample pipeline completed.")


if __name__ == "__main__":
    # Import argparse for command line interface
    import argparse

    parser = argparse.ArgumentParser(description="Lesion Detection and Tracking")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess datasets")
    preprocess_parser.add_argument("--output_dir", required=True, help="Output directory")
    preprocess_parser.add_argument("--ph2_dir", help="PH2 dataset directory")
    preprocess_parser.add_argument("--coco_json", help="COCO annotation JSON file")
    preprocess_parser.add_argument("--coco_images_dir", help="Directory with COCO images")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train_dir", required=True, help="Training data directory")
    train_parser.add_argument("--val_dir", required=True, help="Validation data directory")
    train_parser.add_argument("--model_path", required=True, help="Path to save model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect lesions in an image")
    detect_parser.add_argument("--image_path", required=True, help="Image file path")
    detect_parser.add_argument("--model_path", required=True, help="Path to trained model")
    detect_parser.add_argument("--output_path", required=True, help="Output image path")

    # Track command
    track_parser = subparsers.add_parser("track", help="Track lesions for a patient")
    track_parser.add_argument("--image_path", required=True, help="Image file path")
    track_parser.add_argument("--patient_id", required=True, help="Patient ID")
    track_parser.add_argument("--model_path", required=True, help="Path to trained model")
    track_parser.add_argument("--tracking_dir", required=True, help="Tracking data directory")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze patient progression")
    analyze_parser.add_argument("--patient_id", required=True, help="Patient ID")
    analyze_parser.add_argument("--tracking_dir", required=True, help="Tracking data directory")

    # Match command
    match_parser = subparsers.add_parser("match", help="Match lesions between two timepoints")
    match_parser.add_argument("--patient_id", required=True, help="Patient ID")
    match_parser.add_argument("--timestamp1", required=True, help="First timestamp")
    match_parser.add_argument("--timestamp2", required=True, help="Second timestamp")
    match_parser.add_argument("--tracking_dir", required=True, help="Tracking data directory")

    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate command
    if args.command == "preprocess":
        from dataset_preprocessing import DatasetPreprocessor

        preprocessor = DatasetPreprocessor(args.output_dir)

        files = []
        if args.ph2_dir:
            ph2_files = preprocessor.process_ph2_dataset(args.ph2_dir)
            files.extend(ph2_files)

        if args.coco_json and args.coco_images_dir:
            coco_files = preprocessor.process_coco_dataset(args.coco_json, args.coco_images_dir)
            files.extend(coco_files)

        preprocessor.split_dataset(files)
        preprocessor.visualize_samples()

    elif args.command == "train":
        # In a real implementation, use a proper Mask R-CNN framework
        print("For training Mask R-CNN, use a proper implementation like Matterport's.")
        print("See the instructions in the code for details.")

    elif args.command == "detect":
        # In a real implementation, load the model and perform detection
        print("For detection, use a proper Mask R-CNN implementation.")

    elif args.command == "track":
        # Initialize tracker and track lesions
        tracker = LesionTracker(None, args.tracking_dir)
        result = tracker.detect_and_track_lesions(args.image_path, args.patient_id)
        print(f"Detected {result['lesion_count']} lesions")

    elif args.command == "analyze":
        # Analyze patient progression
        tracker = LesionTracker(None, args.tracking_dir)
        progression = tracker.analyze_progression(args.patient_id)
        print(f"Progression status: {progression['status']}")

    elif args.command == "match":
        # Match lesions between timepoints
        tracker = LesionTracker(None, args.tracking_dir)
        match_result = tracker.match_lesions_between_images(
            args.patient_id, args.timestamp1, args.timestamp2
        )
        print(f"Matched {len(match_result['matched_lesions'])} lesions")
        print(f"New lesions: {match_result['new_lesions_count']}")
        print(f"Disappeared lesions: {match_result['disappeared_lesions_count']}")
