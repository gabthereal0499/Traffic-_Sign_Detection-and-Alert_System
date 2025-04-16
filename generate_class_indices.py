from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Step 1: Path to dataset (relative or full path)
train_data_path = r"C:\Users\user\Downloads\traffic_sign_classification_dataset\train"  # Update this path if needed

# Step 2: Check if path exists
if not os.path.exists(train_data_path):
    print("❌ Dataset path is incorrect.")
    exit()

# Step 3: Create generator
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(train_data_path, target_size=(224, 224))

# Step 4: Save class indices
class_indices = generator.class_indices

with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

print("✅ class_indices saved to class_indices.json")