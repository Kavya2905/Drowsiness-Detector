import streamlit as st
import cv2 
import tempfile
import numpy as np

st.title("""
         LIFE SAVER
        """
        )
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('https://www.pixelstalk.net/wp-content/uploads/2016/06/Light-Blue-HD-Backgrounds-Free-Download.jpg')
    }
    .title{
        color:#E0FFFF
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader('Webcam Live')
run = st.button('Run')

import cv2
import dlib
from scipy.spatial import distance
import time
from time import sleep

FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio


hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
prediction = 0
start_time=time.time()

while run:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y


        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y


        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)


        FRAME_WINDOW.image(frame)
        end_time=time.time()
        elapsed = end_time - start_time

        if EAR<0.26:
            prediction += 1
            if elapsed < 120:
                if prediction > 30:
                    st.write('ALERT! Driver may be drunk...')
                    st.stop()
