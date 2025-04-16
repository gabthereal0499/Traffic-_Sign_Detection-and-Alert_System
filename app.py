# === Import Required Libraries ===
import os
import json
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from twilio.rest import Client

# === Twilio Configuration for SMS Notifications ===
TWILIO_ACCOUNT_SID = "Account_SID"  # Your Twilio Account SID
TWILIO_AUTH_TOKEN = "Auth_Token"      # Your Twilio Auth Token
TWILIO_PHONE_NUMBER = "twilio number"                        # Twilio phone number
DESTINATION_PHONE_NUMBER = "phone number"                  # Receiver’s phone number
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)      # Create Twilio client

# === Flask App Configuration ===
app = Flask(__name__)                                       # Create Flask app
UPLOAD_FOLDER = "static/uploads"                            # Folder to save uploaded images
TRAINING_SAMPLES_FOLDER = "static/training_samples"         # Folder containing training reference images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)                   # Create upload folder if it doesn’t exist
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER                 # Set Flask config for upload folder

# === Load Trained Model and Class Mapping ===
model = load_model("notebooks/traffic_sign_model.h5")       # Load trained model

# Load the mapping from class name to index (saved during training)
with open("web_app/class_indices.json", "r") as f:
    class_indices = json.load(f)

# Create a reverse mapping: index to class name
index_to_class = {v: k for k, v in class_indices.items()}

# === Flask Route: Home Page and Prediction Handling ===
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":                            # Handle POST request (when form is submitted)
        file = request.files["file"]                        # Get the uploaded file
        if file:
            filename = secure_filename(file.filename)       # Sanitize filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)  # Full path to save file
            file.save(filepath)                              # Save uploaded file

            # === Preprocess Image for Prediction ===
            image = load_img(filepath, target_size=(224, 224))    # Load and resize image
            image = img_to_array(image) / 255.0                   # Convert to array and normalize
            image = np.expand_dims(image, axis=0)                 # Add batch dimension

            # === Predict Traffic Sign Class ===
            predictions = model.predict(image)                    # Predict with model
            class_id = int(np.argmax(predictions))                # Get class index with highest probability
            class_name = index_to_class[class_id]                 # Convert index to class name

            # === Send SMS Using Twilio ===
            client.messages.create(
                body=f"Detected Traffic Sign: {class_name}",     # Message content
                from_=TWILIO_PHONE_NUMBER,                       # From Twilio number
                to=DESTINATION_PHONE_NUMBER                      # To user's phone number
            )

            # === Load Reference Image for Predicted Class ===
            class_folder_path = os.path.join(TRAINING_SAMPLES_FOLDER, class_name)  # Folder for predicted class
            ref_img_path = None
            if os.path.exists(class_folder_path):                          # Check if folder exists
                for f in os.listdir(class_folder_path):                    # Loop through files in folder
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):     # Check for image files
                        ref_img_path = os.path.join("training_samples", class_name, f)  # Build relative path
                        break                                              # Use first found image

            # === Return Results to HTML Template ===
            return render_template(
                "index.html", 
                filename=filename, 
                prediction=class_name, 
                reference_image_path=ref_img_path
            )

    # === GET Request: Just Load Form Initially ===
    return render_template(
        "index.html", 
        filename=None, 
        prediction=None, 
        reference_image_path=None
    )

# === Route to Serve Uploaded Image ===
@app.route("/static/uploads/<filename>")
def send_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)  # Serve file from upload folder

# === Start Flask App ===
if __name__ == "__main__":
    app.run(debug=True)                      # Run Flask in debug mode (for development)
