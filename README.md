# CIFAR-10 Web Classifier

This is a web application built using **TensorFlow** and **Streamlit** that classifies images into one of the 10 categories from the **CIFAR-10 dataset**. The app allows users to upload images, and it returns predictions along with probabilities for each class.

## Features
- **Image Upload**: Upload an image in **JPG** or **PNG** format.
- **Image Preprocessing**: Automatically resizes the image to **32x32 pixels** and normalizes it for model input.
- **Prediction Display**: Shows predicted class with probabilities in a bar chart.
- **Model**: Pre-trained **CIFAR-10 classification model** to predict the class of the uploaded image.

## How It Works
1. **Upload an Image**: Users upload an image of their choice in JPG or PNG format.
2. **Preprocessing**: The uploaded image is resized to 32x32 pixels and normalized to the range [0, 1].
3. **Model Prediction**: The pre-trained model classifies the image and outputs predictions for each of the 10 classes.
4. **Result Visualization**: The results are shown in the form of a bar chart, where each class's probability is plotted.

## Requirements
- Python 3.x
- TensorFlow
- Streamlit
- Matplotlib
- Pillow
- NumPy

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SeemaSingh15/CIFAR-10-Web-Classifier.git
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```bash
    streamlit run main.py
    ```

## How to Use
1. Launch the app by running `streamlit run main.py` in your terminal.
2. Upload an image in JPG or PNG format.
3. The app will display the uploaded image along with a bar chart showing the prediction probabilities for each class.

## Model
The model used in this project is a **pre-trained CIFAR-10 classifier** that has been trained on the **CIFAR-10 dataset** containing 60,000 32x32 pixel color images across 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Notes
- The model performs best when images are clear and belong to one of the CIFAR-10 classes.
- Ensure the image is of good quality for accurate predictions.
- The model is trained to handle images of size 32x32 pixels, so larger images are resized automatically.

## Troubleshooting
- **Misclassification**: The model may misclassify images if they are blurry, low quality, or contain objects from multiple classes.
- **Input Format**: Ensure that you upload images in **JPG** or **PNG** format.
- **Prediction Accuracy**: The model's accuracy may drop if the image contains elements not covered by the CIFAR-10 classes.

## Author
Created by **Seema**.

