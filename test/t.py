import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with open("./Inference/inference_args.json", "r") as f:
    inference_config = json.load(f)
selected_columns = inference_config["selected_columns"]

with open("./Inference/character_to_prediction_index.json", "r") as f:
    character_map = json.load(f)
rev_character_map = {v: k for k, v in character_map.items()}

model_path = "./Inference/model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)

def capture_landmarks():
    frames_data = []
    padding_frames = 15

    empty_frame = {col: 0.0 for col in selected_columns}
    for _ in range(padding_frames):
        frames_data.append(empty_frame.copy())

    try:
        cap = cv2.VideoCapture(2)
        with mp_holistic.Holistic(
            min_detection_confidence=0.9, min_tracking_confidence=0.9
        ) as holistic:
            frame = 0
            print_str = ""
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                frame_data = {"frame": frame}

                if results.face_landmarks:
                    for idx, point in enumerate(results.face_landmarks.landmark):
                        frame_data[f"x_face_{idx}"] = point.x
                        frame_data[f"y_face_{idx}"] = point.y
                        frame_data[f"z_face_{idx}"] = point.z

                if results.pose_landmarks:
                    for idx, point in enumerate(results.pose_landmarks.landmark):
                        frame_data[f"x_pose_{idx}"] = point.x
                        frame_data[f"y_pose_{idx}"] = point.y
                        frame_data[f"z_pose_{idx}"] = point.z

                if results.left_hand_landmarks:
                    for idx, point in enumerate(results.left_hand_landmarks.landmark):
                        frame_data[f"x_left_hand_{idx}"] = point.x
                        frame_data[f"y_left_hand_{idx}"] = point.y
                        frame_data[f"z_left_hand_{idx}"] = point.z

                if results.right_hand_landmarks:
                    for idx, point in enumerate(results.right_hand_landmarks.landmark):
                        frame_data[f"x_right_hand_{idx}"] = point.x
                        frame_data[f"y_right_hand_{idx}"] = point.y
                        frame_data[f"z_right_hand_{idx}"] = point.z

                frames_data.append(frame_data)

                print_str = get_str(frames_data)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    image,
                    print_str,
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("MediaPipe Holistic", image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                frame += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

def get_str(frames_data) -> str:
    df = pd.DataFrame(frames_data).fillna(0.0)

    missing_columns = [col for col in selected_columns if col not in df.columns]
    for col in missing_columns:
        df[col] = 0.0

    frames = df[selected_columns].to_numpy(dtype=np.float32)

    input_details = interpreter.get_input_details()

    expected_shape = input_details[0]["shape"]
    if len(frames.shape) != len(expected_shape):
        raise ValueError(
            f"Dimension mismatch: Expected {len(expected_shape)} dimensions, but got {len(frames.shape)}."
        )
    if frames.shape[1] != expected_shape[1]:
        raise ValueError(
            f"Shape mismatch: Model expects {expected_shape[1]} features, but got {frames.shape[1]}."
        )

    interpreter.resize_tensor_input(input_details[0]["index"], frames.shape)
    interpreter.allocate_tensors()

    prediction_fn = interpreter.get_signature_runner("serving_default")
    output = prediction_fn(inputs=frames)

    prediction_str = "".join(
        [rev_character_map.get(s, "") for s in np.argmax(output["outputs"], axis=1)]
    )

    return prediction_str

if __name__ == "__main__":
    capture_landmarks()