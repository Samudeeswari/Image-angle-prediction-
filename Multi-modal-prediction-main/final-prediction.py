# ---------------------------------------------------------------------------------------
# This script combines our trained EMG and Image-based gesture recognition models into a 
# single multimodal prediction system. The ultimate goal is to predict motor angles that 
# will control the movement of our prosthetic leg. 
#
# **Key Notes:**
# - The EMG model's predictions will be prioritized over the Image model's due to its 
#   higher reliability and accuracy. The Image model, while useful, suffers from limited 
#   data quality and quantity due to company restrictions. 
# - The final prediction for the motor angle is determined using a weighted combination 
#   of outputs from both models.
# ---------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

# Load the trained EMG model
emg_model_path = 'Best_model_V0.pt'
emg_model = torch.load(emg_model_path)
emg_model.eval()  # Set to evaluation mode

# Load the trained Image model
image_model_path = 'best_model.keras'
image_model = torch.load(image_model_path)
image_model.eval()  # Set to evaluation mode

# Define weightage for each model
EMG_WEIGHT = 0.7  # EMG predictions have 70% weightage
IMAGE_WEIGHT = 0.3  # Image predictions have 30% weightage

# Define a function to preprocess EMG data
def preprocess_emg_data(raw_emg_signal):
    """
    Preprocess the raw EMG signal data to the format required by the EMG model.
    This function should include any necessary filtering, scaling, or transformation.
    """
    # Example: Normalize the signal (replace this with actual preprocessing steps)
    normalized_signal = (raw_emg_signal - np.mean(raw_emg_signal)) / np.std(raw_emg_signal)
    tensor_signal = torch.tensor(normalized_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, ...)
    return tensor_signal

# Define a function to preprocess Image data
def preprocess_image_data(raw_image):
    """
    Preprocess the raw image data to the format required by the Image model.
    This function should include resizing, normalization, and other transformations.
    """
    # Example: Normalize image (replace with actual preprocessing steps)
    normalized_image = (raw_image - np.mean(raw_image)) / np.std(raw_image)
    tensor_image = torch.tensor(normalized_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
    return tensor_image

# Define a function to predict motor angles using the combined model
def predict_motor_angle(emg_signal, image_data):
    """
    Combine predictions from the EMG and Image models to estimate motor angle.
    """
    # Preprocess the input data
    emg_input = preprocess_emg_data(emg_signal)
    image_input = preprocess_image_data(image_data)

    # Get predictions from both models
    emg_output = emg_model(emg_input)  # Shape: (1, num_classes)
    image_output = image_model(image_input)  # Shape: (1, num_classes)

    # Combine predictions using weighted average
    combined_output = EMG_WEIGHT * emg_output + IMAGE_WEIGHT * image_output

    # Apply softmax to get probabilities (optional, based on model output format)
    combined_probabilities = nn.Softmax(dim=1)(combined_output)

    # Determine the predicted motor angle (or gesture class)
    predicted_class = torch.argmax(combined_probabilities, dim=1).item()
    motor_angle = class_to_motor_angle(predicted_class)  # Convert class to angle
    return motor_angle

# Map gesture classes to motor angles
def class_to_motor_angle(gesture_class):
    """
    Map the predicted gesture class to a motor angle for prosthetic movement.
    Replace with the actual mapping for your system.
    """
    mapping = {
        0: 30,   
        1: 45,   
        2: 60,   
        3: 75,   
        4: 90,   
        5: 120
    }
    return mapping.get(gesture_class, 0)  # Default to 0 if class not found

# Example Usage
if __name__ == "__main__":
    # Simulated inputs
    emg_signal = np.random.randn(800)  # Replace with real EMG signal
    image_data = np.random.randn(64, 64)  # Replace with real image data (e.g., 64x64)

    # Get motor angle prediction
    motor_angle = predict_motor_angle(emg_signal, image_data)
    print(f"Predicted Motor Angle: {motor_angle} degrees")
