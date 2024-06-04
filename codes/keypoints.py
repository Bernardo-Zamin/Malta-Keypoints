import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import ast


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.6
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.6
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def process_frame_for_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(frame_rgb)
    
    black_frame = np.zeros(frame.shape, dtype=np.uint8)
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)

    mp_drawing.draw_landmarks(black_frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, drawing_spec, drawing_spec)
    mp_drawing.draw_landmarks(black_frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)
    mp_drawing.draw_landmarks(black_frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)

    return black_frame

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        boca_ids = [61, 291, 13, 14, 0]
        for face_landmarks in face_results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):
                if id in boca_ids:
                    ih, iw, _ = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    return frame

def process_videos_from_csv(df):
    for index, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row['Paths'].strip()

        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo: {video_path}")
            continue
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start_frame = 0
            middle_frame1 = total_frames // 3
            middle_frame2 = total_frames // 3 + 5
            end_frame = total_frames - 1

            frames = get_specific_frames(
                cap, [start_frame, middle_frame1, middle_frame2, end_frame])
            keypoint_frames = [process_frame_for_keypoints(
                frame) for frame in frames]
            final_image = create_final_image(frames, keypoint_frames)

            video_folder_name = video_path.split('/')[-3]
            namex = video_path.split('/')[-1].split('.')[0]
            final_name = f"{video_folder_name} - {namex}"

            output_image_path = f'/mnt/B-SSD/bernardo/output/{final_name}.jpg'
            cv2.imwrite(output_image_path, final_image)
        except Exception as e:
            print(f"Erro ao processar vídeo {video_path}: {e}")
        finally:
            cap.release()

def get_specific_frames(cap, frame_indices):
    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames

def create_final_image(frames, processed_frames):
    top = np.concatenate(frames, axis=1)
    bottom = np.concatenate(processed_frames, axis=1)
    final_image = np.concatenate([top, bottom], axis=0)
    final_image = cv2.resize(final_image, (1920, 1080))
    return final_image

# Carregar o CSV e processar os vídeos
# csv_path = '/mnt/B-SSD/bernardo/csvs/Paths_only.csv'
# df = pd.read_csv(csv_path)
# process_videos_from_csv(df)