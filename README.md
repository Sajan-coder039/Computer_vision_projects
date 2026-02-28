# Computer_vision_projects
# 😷 Real-Time Face Mask Detector

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

A computer vision application that detects in real-time whether a person is wearing a face mask using a custom-trained Convolutional Neural Network (CNN) and OpenCV.

## 🚀 Overview
This project captures live video feed from a webcam and processes each frame through a Deep Learning model to classify the user's status. It was developed to explore the integration of **PyTorch** models within **OpenCV** pipelines.



## 🧠 Technical Workflow
1. **Data Collection:** Images were unzipped and pre-processed in Google Colab.
2. **Model Training:** A custom CNN architecture was trained to distinguish between 'Mask' and 'No Mask' classes.
3. **Inference:**
   * Captured frames via `cv2.VideoCapture`.
   * Normalized and converted BGR frames to PyTorch Tensors.
   * Performed CPU-based inference using `map_location`.
4. **Visualization:** Real-time text overlays on the video stream.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sajan-coder039/Computer_vision_projects.git](https://github.com/Sajan-coder039/Computer_vision_projects.git)
   cd Computer_vision_projects
