# Face Recognition System

This project is a Python-based face recognition system using OpenCV. It utilizes Haar cascade classifiers and LBPH (Local Binary Patterns Histograms) face recognition for detecting and recognizing faces in images and real-time webcam feeds.

## Features

- Detects and recognizes faces in images.
- Performs real-time face detection and recognition through webcam feed.
- Provides confidence levels for face recognition.
- Rescales input frames for better processing.
- Handles errors gracefully when necessary files are not found.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/gaikwadyash905/Face-Detection-App.git
```

2. Install the required dependencies:
```bash
pip install opencv-python
pip install numpy
```
## Usage
1. Ensure you have:

  * Trained Haar cascade classifier (```haarcascade_frontalface_alt.xml```).
  * Trained LBPH face recognizer model (face_trained.yml).
  * Training images of faces stored in the ```training_faces/``` directory.
  * deleted ```new``` file from every ```Person``` folder

2. Modify the file paths in the scripts (face_train.py and face_recognizer.py) according to your directory structure.

3. Run the face training script to train the face recognition model:

```bash 
python face_train.py
```
4. Once the model is trained, run the face recognition script to detect and recognize faces:
```bash
python face_recognizer.py
```

5. Optionally, you can uncomment the lines in the scripts to perform face detection and recognition on a single image.

6. If you want to use specific images from the directory instead of all of them, you can uncomment the array and specify the image paths directly instead of using the loop in the face_train.py script.

# Contributing
Contributions are welcome! If you'd like to contribute to this project, please open an issue to discuss the changes you'd like to make or submit a pull request.
