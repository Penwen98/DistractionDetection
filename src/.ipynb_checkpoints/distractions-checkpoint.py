import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,54,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(categories), activation='softmax'))

model.load_weights(sys.path[0] + '/modelweights.h5')


def webcamSetup():
    global distr_dict, cap

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label a state
    distr_dict = {0: "Closed eyes", 1: "Safe driving", 2: "Smoking", 3: "Talking phone", 4: "Texting phone", 5: "Turning"}
    
    # start the webcam feed
    cap = cv2.VideoCapture(0)

class NoFaceDetectedException(Exception):
    def msg():
        return "No face detected on the screen"

def getDistraction():
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        return 0
    
    facecasc = cv2.CascadeClassifier(sys.path[0] + '/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.05, minNeighbors=5)
    
    prediction = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (54, 128)), -1), 0)
        prediction = model.predict(cropped_img)
    
    data = {}
    if type(prediction) is np.ndarray:
        for distraction in distr_dict:
            data[distr_dict[distraction]] = round(prediction[0][distraction].item(), 3)
    else:
        raise NoFaceDetectedException
    
    return data