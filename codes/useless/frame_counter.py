import os
from tqdm import tqdm 
import cv2

# Conta quantos frames tem em cada video do diretorio

def fps_counter(videos_path):
    video_files = [file for file in os.listdir(videos_path) if file.endswith(('.mp4', '.avi', '.mov'))]
    total = 0
    for video_file in video_files:
        video_path = os.path.join(videos_path, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo: {video_path}")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total += total_frames
        # print(f"O video {video_file} tem {total_frames} frames")
        cap.release()
    print("\n A media de frames do dataset é de : ", total/len(video_files), "frames")


videos_path = '/mnt/B-SSD/LIBRAS-dataset/datasets/youtube_v2/videos'

fps_counter(videos_path)


