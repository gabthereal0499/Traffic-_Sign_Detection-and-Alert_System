import os
import shutil
from pathlib import Path

# Source: your training dataset
SOURCE_DIR = r"C:\Users\user\Downloads\traffic_sign_classification_dataset\train"

# Destination: where Flask can serve training images
DEST_DIR = r"C:\Users\user\Desktop\Traffic_sign_detection_project\web_app\static\training_samples"

# Create destination if not exists
os.makedirs(DEST_DIR, exist_ok=True)

# Loop through class folders
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if os.path.isdir(class_path):
        dest_class_path = os.path.join(DEST_DIR, class_name)
        os.makedirs(dest_class_path, exist_ok=True)

        # Get first image from the class folder
        for file in os.listdir(class_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                src_file = os.path.join(class_path, file)
                dst_file = os.path.join(dest_class_path, file)
                shutil.copyfile(src_file, dst_file)
                print(f"Copied {file} to {dest_class_path}")
                break  # Only 1 image per class
