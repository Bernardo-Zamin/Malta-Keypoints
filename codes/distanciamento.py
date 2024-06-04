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


def get_specific_frames(cap, frame_indices):
    frames = []
    for frame_index in frame_indices:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        except Exception as e:
            print(f"Erro ao ler o frame {frame_index}: {e}")
    return frames


def resize_image(image, target_size):
    """ Redimensiona a imagem para o tamanho-alvo com preservação de aspecto e preenchimento se necessário. """

    ih, iw = image.shape[:2]
    th, tw = target_size
    scale = min(tw/iw, th/ih)

    nh, nw = int(ih * scale), int(iw * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    top, bottom = (th - nh) // 2, (th - nh + 1) // 2
    left, right = (tw - nw) // 2, (tw - nw + 1) // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def create_final_image(frames, processed_frames, target_size=(1280, 720)):
    try:

        resized_frames = [resize_image(frame, target_size) for frame in frames]
        resized_processed_frames = [resize_image(
            frame, target_size) for frame in processed_frames]

        top = np.concatenate(resized_frames, axis=1)
        bottom = np.concatenate(resized_processed_frames, axis=1)
        final_image = np.concatenate([top, bottom], axis=0)
    except Exception as e:
        print(f"Erro ao criar a imagem final: {e}")
        final_image = np.zeros((*target_size, 3), dtype=np.uint8)

    return final_image


def modificar_distancia(keypoints, factor):
    if not keypoints:
        return []

    keypoints = np.array(keypoints)
    centro = np.mean(keypoints, axis=0)
    keypoints_modificados = []

    for kp in keypoints:
        direcao = kp - centro
        kp_modificado = centro + direcao * factor
        keypoints_modificados.append(kp_modificado)

    return keypoints_modificados


def create_combined_image(original_frame, keypoint_frame, factor=1.2):

    frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    keypoints = [(landmark.x, landmark.y)
                 for landmark in results.pose_landmarks.landmark] if results.pose_landmarks else []

    modified_keypoints = modificar_distancia(keypoints, factor)

    modified_frame = original_frame.copy()
    for kp in modified_keypoints:
        x, y = int(kp[0] * original_frame.shape[1]
                   ), int(kp[1] * original_frame.shape[0])
        cv2.circle(modified_frame, (x, y), 5, (255, 0, 0), -1)

    combined_frame = np.hstack(
        (original_frame[:, :original_frame.shape[1]//2], modified_frame[:, original_frame.shape[1]//2:]))

    return combined_frame


def process_frame_for_keypoints(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(frame_rgb)

    black_frame = np.zeros(frame.shape, dtype=np.uint8)

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3)

    mp_drawing.draw_landmarks(black_frame, holistic_results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS, drawing_spec, drawing_spec)
    mp_drawing.draw_landmarks(black_frame, holistic_results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)
    mp_drawing.draw_landmarks(black_frame, holistic_results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, drawing_spec, drawing_spec)

    return black_frame


def process_videos_from_csv(df):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        video_paths = ast.literal_eval(row['Paths'])

        for video_path in video_paths:
            cap = None
            try:

                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue

                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    print(f"Erro ao abrir o vídeo: {video_path}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                start_frame = 0
                middle_frame1 = total_frames // 3
                middle_frame2 = total_frames // 3 + 5
                end_frame = total_frames - 1

                frames = get_specific_frames(
                    cap, [start_frame, middle_frame1, middle_frame2, end_frame])

                combined_images = [create_combined_image(
                    frame, process_frame_for_keypoints(frame), factor=1.2) for frame in frames]

                final_image = create_final_image(
                    frames, combined_images, target_size=(1280, 720))

                output_path = f"/mnt/B-SSD/bernardo/aumentados/{os.path.basename(video_path)}.png"
            except Exception as e:
                print(f"Erro ao processar o vídeo {video_path}: {e}")
            finally:
                if cap:
                    cap.release()


csv_path = '/mnt/B-SSD/bernardo/csvs/Paths_only.csv'
df = pd.read_csv(csv_path)
process_videos_from_csv(df)
