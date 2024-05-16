import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Compose, Resize, ToTensor
import requests
from PIL import Image
from io import BytesIO




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

gender_dict = {'female': 0, 'male': 1, 'boy':1, 'girl':0}

transform=Compose([
    Resize((224, 224)),  # Example resize
    ToTensor(),  # Convert images to Tensor
])

Segmentation_model_path = "../models/deeplabv3.tflite"
# Create the options that will be used for ImageSegmenter
base_options = python.BaseOptions(model_asset_path=Segmentation_model_path)
options = vision.ImageSegmenterOptions(base_options=base_options,
                                        output_category_mask=True)

class ChestWidthPredictor(nn.Module):
    def __init__(self):
        super(ChestWidthPredictor, self).__init__()
        # Load a pretrained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])


        # Add additional layers for our specific task
        # Assuming resnet outputs 512 features, we have 2 images, 1 gender, and 1 height as inputs
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 + 2, 256), # 512 for each image, 1 for gender, 1 for height
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1) # Output a single value for chest width
        )

    def forward(self, front_image, side_image, gender, height):
        # Assume images are already tensors and gender/height are normalized
        front_features = self.resnet(front_image).view(front_image.size(0), -1)
        side_features = self.resnet(side_image).view(side_image.size(0), -1)

        # Combine all features
        combined_features = torch.cat((front_features, side_features, gender.unsqueeze(1), height.unsqueeze(1)), dim=1)

        # Pass through the classifier
        output = self.classifier(combined_features)
        return output


def load_image_from_url(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        print(response.status_code)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))
            return image
            # arr = np.frombuffer(response.content, np.uint8)
            # image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            # return image
        else:
            return None
    except Exception as e:
        return None
    
def get_image_from_path(path):
    try:
        im = Image.open(path)
        return im
    except Exception as e:
        return None
        


def segmentation(image):
    BG_COLOR = (0, 0, 0) # gray
    MASK_COLOR = (255, 255, 255) # white

    # Create the image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:

        # Create the MediaPipe image file that will be segmented
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Retrieve the masks for the segmented image
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask

        # Generate solid color images for showing the output segmentation mask.
        image_data = image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

        return output_image

def estimate_chest_width(image,side = 'front'):
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    image_height, image_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if side == 'front':
      if results.pose_landmarks:
          shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
          shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
          chest_width_pixels = calculate_distance(shoulder_left, shoulder_right) * image_width
          return chest_width_pixels, image_height, image_width
      else:
          print("Pose landmarks not detected.")
          return 0, image_height, image_width
    if side == 'side':
      if results.pose_landmarks:
        # Assuming the chest keypoint corresponds to approximately the midpoint between the shoulders
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        chest_keypoint = ((shoulder_left.x + shoulder_right.x) * 0.5 * image.shape[1],
                          (shoulder_left.y + shoulder_right.y) * 0.5 * image.shape[0])
        return chest_keypoint

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def estimate_chest_measurements(side_mask, chest_keypoint_side):
    """Estimate chest width and depth from segmented images."""
    chest_level_y = int(chest_keypoint_side[1])
    chest_depth_pixels = np.sum(side_mask[chest_level_y] > 0)
    return chest_depth_pixels

def calculate_height_from_segmentation(segmented_mask):
    """Calculate the person's height in pixels from the segmentation mask."""
    person_pixels_y = np.where(segmented_mask > 0)[0]
    if person_pixels_y.size == 0:
        return 0
    height_pixels = person_pixels_y.max() - person_pixels_y.min()
    return height_pixels

def pixels_to_cm(pixel_measure, person_height_pixels, person_height_cm):
    # Calculate conversion factor from pixels to centimeters
    conversion_factor = person_height_cm / person_height_pixels
    return pixel_measure * conversion_factor