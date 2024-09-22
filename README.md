# Censor-Face-From-Video
This project uses OpenCV's Deep Neural Network (DNN) module to detect faces from a live webcam feed. Once a face is detected, it is blurred using a Gaussian blur filter instead of drawing a bounding box. The code is designed for real-time face detection and applies blurring to each detected face.

#Requirements
OpenCV: Install the OpenCV library (which includes the cv2.dnn module) using:
#pip install opencv-python opencv-python-headless

Pre-trained DNN Model: The model used is based on a Single Shot Multibox Detector (SSD) and is pre-trained on face detection.

deploy.prototxt: Defines the architecture of the network.
res10_300x300_ssd_iter_140000.caffemodel: Contains the pre-trained weights for the face detection model.
These files are present in the project directory.

NumPy: Ensure NumPy is installed, which is required for numerical operations:
pip install numpy


#How the Code Works
Load the DNN Model:

The face detection model is loaded using cv2.dnn.readNetFromCaffe(), which reads the model architecture and weights.
Capture Webcam Input:

The webcam feed is captured using cv2.VideoCapture(0).
Pre-process the Frame:

Each frame is resized to 300x300 pixels and converted into a "blob" format using cv2.dnn.blobFromImage(). This blob is the required input format for the DNN model.
Perform Face Detection:

The model processes the blob and returns detections. Each detection is a face with an associated confidence score.
Only faces with confidence greater than 0.5 are considered.
Blur the Detected Face:

For each detected face, the bounding box coordinates are calculated, and a Gaussian blur is applied to the face region using cv2.GaussianBlur().
Display the Result:

The resulting frame, with the blurred face, is displayed in a window.
The loop continues until the user presses the 'q' key to exit.


Run the Script: Simply run the script using Python:
python face_blur.py
Quit: Press the 'q' key to exit the video stream and close the window.

Customization
Adjusting the Blur: You can increase or decrease the strength of the blur by modifying the kernel size and sigma value in the cv2.GaussianBlur() function:
blurred_face = cv2.GaussianBlur(face_region, (99, 99), 60)
Larger kernel sizes like (151, 151) will increase the blur effect.

Threshold for Confidence: The current confidence threshold for face detection is set to 0.5. You can adjust this value to make detection more or less strict.

Output
The program will open a window showing your webcam feed with detected faces blurred in real-time.
