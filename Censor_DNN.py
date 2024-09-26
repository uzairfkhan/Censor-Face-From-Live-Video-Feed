import cv2
import numpy as np
def main():
    # Load the DNN model for face detection
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get the height and width of the frame
        h, w = frame.shape[:2]

        # Create a blob from the frame to feed into the DNN model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Set the input blob to the network
        net.setInput(blob)

        # Perform face detection
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Only consider detections with confidence > 0.5
            if confidence > 0.5:
                # Get the bounding box coordinates of the detected face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Make sure the coordinates are within the frame size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Extract the face region and apply a Gaussian blur
                face_region = frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 60)
                frame[y1:y2, x1:x2] = blurred_face

        # Display the resulting frame with blurred faces
        cv2.imshow('DNN Face Detection with Blurred Faces', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
