import tkinter
import cv2
import numpy as np
from tkinter import Label, Tk, Scale, Button, HORIZONTAL
from PIL import Image, ImageTk
import mediapipe as mp
import os


def resize_keypoints(keypoints, scale_factor):
    center = np.mean(keypoints, axis=0)
    return scale_factor * (keypoints - center) + center

def update_frame(scale_val):
    global resized_frame
    scale_factor = scale_val / 100.0
    resized_kps = resize_keypoints(keypoints, scale_factor)
    resized_frame = draw_skeleton(frame.copy(), resized_kps)
    display_image(resized_frame)

def draw_skeleton(frame, keypoints):
    
    return frame

def save_frame():
    cv2.imwrite("resized_keypoints/saved_frame.jpg", resized_frame)

def display_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.config(image=imgtk)

def generate_holistic_keypoints(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic = mp.solutions.holistic.Holistic()
    results = holistic.process(rgb_frame)

    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.append((lm.x, lm.y))

    
    return keypoints

def draw_skeleton(frame, keypoints):
    for kp in keypoints:
        x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  
    return frame

def update_frame(scale_val):
    global resized_frame, keypoints
    scale_factor = scale_val / 100.0
    resized_kps = resize_keypoints(np.array(keypoints), scale_factor)
    resized_frame = draw_skeleton(frame.copy(), resized_kps)
    display_image(resized_frame)


if 'DISPLAY' in os.environ:
    root = tkinter.Tk()
    
else:
    print("No display available")


frame = cv2.imread('/mnt/B-SSD/bernardo/codes/frame/frame.png')
keypoints = [...]  


root = Tk()
scale = Scale(root, from_=50, to=150, orient=HORIZONTAL, command=update_frame)
scale.pack()
button = Button(root, text="Salvar", command=save_frame)
button.pack()
label = Label(root)
label.pack()


resized_frame = frame.copy()
display_image(resized_frame)


root.mainloop()
