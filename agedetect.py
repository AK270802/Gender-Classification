from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load the pre-trained gender detection model
model = load_model('gender_detection.model')

# Open the video file
video_path = 'IMG_0947.MOV'
video = cv2.VideoCapture(video_path)

classes = ['man', 'woman']

# Get the video dimensions
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create an output window and resize it
cv2.namedWindow("Gender Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gender Detection", width, height)

# Read the video frames
while video.isOpened():
    # Read a frame from the video
    status, frame = video.read()

    # Break the loop if the video has ended
    if not status:
        break

    # Apply face detection
    faces, confidences = cv.detect_face(frame)

    # Loop through the detected faces
    for face, confidence in zip(faces, confidences):
        # Get the corner points of the face rectangle
        startX, startY, endX, endY = face

        # Draw a rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        # Perform preprocessing on the cropped face image
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on the preprocessed face image
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write the label and confidence above the face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Gender Detection", frame)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
