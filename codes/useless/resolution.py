import pandas as pd
import cv2
import ast

csv_file_path = '/mnt/B-SSD/marcelo/updated_definitive_words_and_paths.csv'
data = pd.read_csv(csv_file_path)

def get_video_resolution(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height
    except Exception as e:
        print(f"Erro ao abrir vídeo {video_path}: {e}")
        return None, None

resolutions = []

for index, row in data.iterrows():
    video_paths = ast.literal_eval(row['paths'])
    for path in video_paths:
        w, h = get_video_resolution(path)
        if w is not None and h is not None:
            resolutions.append((w, h))

if resolutions:
    avg_width = sum(w for w, h in resolutions) / len(resolutions)
    avg_height = sum(h for w, h in resolutions) / len(resolutions)
    avg_resolution = (avg_width, avg_height)
else:
    avg_resolution = None

print(avg_resolution)

