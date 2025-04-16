# # # from keras.models import model_from_json
# # # import cv2
# # # import numpy as np

# # # # Load the model architecture from JSON file
# # # json_file = open("model-bw.json", "r")
# # # model_json = json_file.read()
# # # json_file.close()

# # # # Create the model from the loaded architecture
# # # model = model_from_json(model_json)

# # # # Load the model weights
# # # model.load_weights("model-bw.weights.h5")

# # # # Function to extract features from an image
# # # def extract_features(image):
# # #     feature = np.array(image)
# # #     feature = feature.reshape(1, 48, 48, 1)
# # #     return feature / 255.0

# # # # Define the labels
# # # label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'j', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# # # # Open a video capture device (webcam)
# # # cap = cv2.VideoCapture(0)

# # # # Main loop to capture frames and perform prediction
# # # while True:
# # #     # Capture frame-by-frame
# # #     ret, frame = cap.read()

# # #     # If frame could not be captured, break the loop
# # #     if not ret:
# # #         print("Error: Unable to capture frame")
# # #         break

# # #     # Draw a rectangle on the frame for cropping
# # #     cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)

# # #     # Crop the frame
# # #     cropframe = frame[40:300, 0:300]

# # #     # If cropframe is None, break the loop
# # #     if cropframe is None:
# # #         print("Error: Unable to crop frame")
# # #         break

# # #     # Convert cropframe to grayscale and resize
# # #     cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
# # #     cropframe = cv2.resize(cropframe, (48, 48))

# # #     # Extract features from cropframe
# # #     cropframe = extract_features(cropframe)

# # #     # Make prediction using the model
# # #     pred = model.predict(cropframe)

# # #     # Get the index of the maximum value in pred
# # #     pred_index = np.argmax(pred)

# # #     # Check if the index is within bounds of the label list
# # #     if pred_index < len(label):
# # #         prediction_label = label[pred_index]
# # #     else:
# # #         prediction_label = "Unknown"  # Handle out-of-bounds index gracefully

# # #     # Draw a rectangle for displaying the prediction
# # #     cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)

# # #     # Display prediction on the frame
# # #     if prediction_label == 'Unknown':
# # #         cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
# # #     else:
# # #         accu = "{:.2f}".format(np.max(pred) * 100)
# # #         cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
# # #                     cv2.LINE_AA)

# # #     # Display the frame
# # #     cv2.imshow("output", frame)

# # #     # Break the loop if 'q' key is pressed
# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # # Release the video capture device and close all OpenCV windows
# # # cap.release()
# # # cv2.destroyAllWindows()



# text to speech feature included 

import cv2
import numpy as np
from keras.models import model_from_json
import pyttsx3
import time

# Initialize the speech engine
engine = pyttsx3.init()

# Load the model architecture from JSON file
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()

# Create the model from the loaded architecture
model = model_from_json(model_json)

# Load the model weights
model.load_weights("model-bw.weights.h5")

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Define the labels
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'j', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize previous prediction and time
prev_prediction = None
prev_time = time.time()

# Main loop to capture frames
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame could not be captured, break the loop
    if not ret:
        print("Error: Unable to capture frame")
        break

    # Draw a rectangle on the frame for cropping
    cv2.rectangle(frame, (0, 40), (320, 320), (148, 0, 211), 3)

    # Crop the frame
    cropframe = frame[40:320, 0:320]

    # If cropframe is None, break the loop
    if cropframe is None:
        print("Error: Unable to crop frame")
        break

    # Convert cropframe to grayscale and resize
    cropframe_gray = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe_gray = cv2.resize(cropframe_gray, (48, 48))

    # Extract features from cropframe
    cropframe_features = extract_features(cropframe_gray)

    # Make prediction using the model
    pred = model.predict(cropframe_features)

    # Get the index of the maximum value in pred
    pred_index = np.argmax(pred)

    # Check if the index is within bounds of the label list
    if pred_index < len(label):
        prediction_label = label[pred_index]
    else:
        prediction_label = "Unknown"  # Handle out-of-bounds index gracefully

    # Display the predicted text on the crop frame border
    accu = "{:.2f}".format(np.max(pred) * 100)
    text_to_display = f'Predicted alphabet: {prediction_label}, Accuracy: {accu}%'
    cv2.putText(frame, text_to_display, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("output", frame)

    # Calculate time difference
    current_time = time.time()
    time_diff = current_time - prev_time

    # Speak if the prediction remains the same for more than 10 seconds
    if prediction_label == prev_prediction and time_diff > 10:
        # Speak the predicted alphabet
        engine.say(text_to_display)
        engine.runAndWait()

        # Update the previous time
        prev_time = current_time

    # Update the previous prediction
    prev_prediction = prediction_label

    # Break the loop if 'q' or 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 'q' or 'Esc' key
        break

# Release the video capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
