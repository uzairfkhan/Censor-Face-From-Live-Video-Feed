import cv2
import os
import numpy as np


def process_video(input_path, output_path, net):
    # Open the input video
    cap = cv2.VideoCapture(input_path)

    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the input video
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Set the output video writer with the same properties as the input video
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get the height and width of the frame
        h, w = frame.shape[:2]

        # Create a blob from the frame to feed into the DNN model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)

        # Perform face detection
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.45:
                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # Ensure the bounding box is within the frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Blur the detected face
                face_region = frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 60)
                frame[y1:y2, x1:x2] = blurred_face

        # Write the frame to the output video
        out.write(frame)

    # Release the capture and writer
    cap.release()
    out.release()


def main():
    # Load the DNN model for face detection
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    # Define input and output directories
    input_dir = 'Raw'
    output_dir = 'Blurred'

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each video in the input directory
    for video_file in os.listdir(input_dir):
        if video_file.endswith('.mp4'):  # Only process .mp4 files
            input_path = os.path.join(input_dir, video_file)
            output_path = os.path.join(output_dir, f"blurred_{video_file}")

            print(f"Processing {video_file}...")
            process_video(input_path, output_path, net)
            print(f"Saved blurred video as {output_path}")


if __name__ == '__main__':
    main()
