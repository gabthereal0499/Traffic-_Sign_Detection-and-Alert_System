# 🚦 Traffic Sign Detection Web App with SMS Alerts

This is a web-based Traffic Sign Detection system that identifies Indian traffic signs from uploaded images using a trained deep learning model. It also sends the predicted traffic sign name via SMS using **Twilio**.

---

## 🔧 Features

- Upload and classify Indian traffic signs using a trained model.
- Displays both the uploaded image and a reference training image for the predicted class.
- Sends SMS alert with the detected sign name using **Twilio**.
- Built with Flask, TensorFlow (MobileNetV2), HTML/CSS, and Twilio API.

---

## 🧠 Model Info

- Model architecture: **MobileNetV2**
- Input image size: `224x224`
- Dataset: 85 Indian traffic sign classes, ~4000 images per class
- Accuracy: Trained and evaluated on augmented data using `ImageDataGenerator`.

---

## 🗂️ Project Structure


---

## 🚀 How to Run

### 1. Clone the repo


git clone https://github.com/your-username/traffic-sign-detection-app.git
cd traffic-sign-detection-app


Install Dependencies

pip install -r requirements

Example dependencies:


Flask
tensorflow
numpy
twilio

3. Add Twilio Configuration
In app.py, replace these values with your Twilio credentials:

TWILIO_ACCOUNT_SID = "your_sid"
TWILIO_AUTH_TOKEN = "your_token"
TWILIO_PHONE_NUMBER = "+1XXXXXX"
DESTINATION_PHONE_NUMBER = "+91XXXXXXXXXX"

4. Run the Web App

python app.py
Then open your browser and visit:
📍 http://127.0.0.1:5000/

NOTES:

Ensure static/training_samples/ has a folder for each class name and at least one image inside.

Make sure the class names match exactly with those in class_indices.json.



