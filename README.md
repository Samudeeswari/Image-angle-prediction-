# Multimodal Prediction System for Prosthetic Motor Angles

## Overview
This repository implements a multimodal prediction system that combines predictions from trained EMG and Image-based gesture recognition models to estimate motor angles for controlling a prosthetic leg. The integration leverages the strengths of both models, with a weighted combination of their outputs determining the final motor angle prediction. This approach enables dynamic and adaptive movement control for prosthetic devices.

## Key Features
- **Priority Weighting:**
  - The EMG model's predictions are given higher weight due to its superior reliability and accuracy.
  - The Image model provides supplementary predictions but is less prioritized due to limited data quality and quantity.
- **Weighted Combination:**
  - Outputs from the EMG and Image models are combined using a weighted average formula to produce the final motor angle.

## Architecture
1. **EMG Model:**
   - A trained PyTorch-based model (`Best_model_V0.pt`) for classifying gestures from EMG signals.
2. **Image Model:**
   - A trained Keras-based model (`best_model.keras`) for predicting motor angles based on images.
3. **Prediction Logic:**
   - Weighted combination of outputs:
     - EMG Model Weight: **0.7**
     - Image Model Weight: **0.3**

## File Description
- **`emg_model_path`:** Path to the trained EMG model.
- **`image_model_path`:** Path to the trained Image model.
- **`preprocess_emg_data(raw_emg_signal)`:** Preprocesses raw EMG data for the EMG model.
- **`preprocess_image_data(raw_image)`:** Preprocesses raw image data for the Image model.
- **`predict_motor_angle(emg_signal, image_data)`:** Combines predictions from both models and determines the final motor angle.
- **`class_to_motor_angle(gesture_class)`:** Maps gesture classes to specific motor angles.

## Workflow
### 1. Data Preprocessing
#### EMG Data
- Normalized using z-score normalization.
- Converted to a tensor format required by the EMG model.

#### Image Data
- Normalized to mean 0 and standard deviation 1.
- Resized and converted to a tensor format suitable for the Image model.

### 2. Prediction Logic
1. **Preprocessing:**
   - Input EMG signal and image data are preprocessed separately.
2. **Model Predictions:**
   - The EMG model predicts gesture classes based on EMG input.
   - The Image model predicts gesture classes or motor angles based on image input.
3. **Weighted Combination:**
   - The outputs from both models are combined using the formula:
     ```
     combined_output = (EMG_WEIGHT * emg_output) + (IMAGE_WEIGHT * image_output)
     ```
4. **Final Prediction:**
   - A softmax layer converts the combined output into probabilities.
   - The class with the highest probability is mapped to a motor angle using `class_to_motor_angle()`.

### 3. Mapping Gesture Classes to Motor Angles
The following mapping converts gesture classes to motor angles:
| Gesture Class | Motor Angle (degrees) |
|---------------|------------------------|
| 0             | 30                     |
| 1             | 45                     |
| 2             | 60                     |
| 3             | 75                     |
| 4             | 90                     |
| 5             | 120                    |

## Example Usage
```python
if __name__ == "__main__":
    # Simulated inputs
    emg_signal = np.random.randn(800)  # Replace with real EMG signal
    image_data = np.random.randn(64, 64)  # Replace with real image data (e.g., 64x64)

    # Get motor angle prediction
    motor_angle = predict_motor_angle(emg_signal, image_data)
    print(f"Predicted Motor Angle: {motor_angle} degrees")
```

## Requirements
- **Python 3.8+**
- **PyTorch**
- **Keras (TensorFlow Backend)**
- **NumPy**

Install dependencies using:
```bash
pip install torch tensorflow numpy
```

## Limitations
- **Data Quality:** The Image model is limited by low-quality data, which affects its performance.
- **Prototype:** This system is a proof-of-concept and may require further fine-tuning and real-world testing for deployment.

## Future Work
1. **Improved Data Collection:**
   - Collect higher-quality EMG and image data to enhance model training and performance.
2. **Real-Time Integration:**
   - Implement real-time prediction pipelines for prosthetic control.
3. **Enhanced Model Architectures:**
   - Experiment with more advanced multimodal architectures (e.g., combining CNNs and RNNs).
4. **Dynamic Weight Adjustment:**
   - Explore adaptive weighting schemes based on model confidence or input conditions.

By addressing these areas, the system can evolve into a reliable tool for prosthetic control, improving mobility and comfort for users.

