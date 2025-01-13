import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe Holistic model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def capture_landmarks(sequence_id, num_frames=100):
    """
    Capture landmarks and return them in the ASL Signs wide format.
    """
    # Initialize an empty list to store frame data
    frames_data = []

    # Open the webcam and process frames
    try:
        cap = cv2.VideoCapture(0)  # Open the webcam
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            frame = 0
            while cap.isOpened() and frame < num_frames:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Prepare the image for processing
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Store landmarks in a dictionary
                frame_data = {'sequence_id': sequence_id, 'frame': frame}

                # Extract face landmarks
                if results.face_landmarks:
                    for idx, point in enumerate(results.face_landmarks.landmark):
                        frame_data[f"x_face_{idx}"] = point.x
                        frame_data[f"y_face_{idx}"] = point.y
                        frame_data[f"z_face_{idx}"] = point.z
                # Extract pose landmarks
                if results.pose_landmarks:
                    for idx, point in enumerate(results.pose_landmarks.landmark):
                        frame_data[f"x_pose_{idx}"] = point.x
                        frame_data[f"y_pose_{idx}"] = point.y
                        frame_data[f"z_pose_{idx}"] = point.z
                # Extract left hand landmarks
                if results.left_hand_landmarks:
                    for idx, point in enumerate(results.left_hand_landmarks.landmark):
                        frame_data[f"x_left_hand_{idx}"] = point.x
                        frame_data[f"y_left_hand_{idx}"] = point.y
                        frame_data[f"z_left_hand_{idx}"] = point.z
                # Extract right hand landmarks
                if results.right_hand_landmarks:
                    for idx, point in enumerate(results.right_hand_landmarks.landmark):
                        frame_data[f"x_right_hand_{idx}"] = point.x
                        frame_data[f"y_right_hand_{idx}"] = point.y
                        frame_data[f"z_right_hand_{idx}"] = point.z

                # Append frame data to the list
                frames_data.append(frame_data)

                # Show the video feed for visualization
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
                    break

                frame += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Convert list of frames into a DataFrame
    landmark_data = pd.DataFrame(frames_data).fillna(np.nan)
    return landmark_data

# Main script
if __name__ == "__main__":
    # Set sequence ID and capture landmarks
    sequence_id = 1209576923  # Example sequence ID
    num_frames = 150  # 10 seconds of capture at 15 FPS

    # Capture landmarks in the required format
    landmark_data = capture_landmarks(sequence_id, num_frames)

    # Save to Parquet file
    output_file = "output_landmarks.parquet"
    landmark_data.to_parquet(output_file)
    print(f"Landmarks saved to {output_file}")
