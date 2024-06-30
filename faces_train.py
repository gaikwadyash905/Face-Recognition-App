import os
import cv2 as cv
import numpy as np

# people = ["Person 1", "Person 2", "Person 3", "Person 4"]

people = []
for i in os.listdir(r"training_faces/"):
  people.append(i)

print("Training faces...")

haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml")

DIR = "training_faces/"

features = []
labels = []

def create_train():
  for person in people:
    path = os.path.join(DIR, person)
    label = people.index(person)
    
    for img in os.listdir(path):
      img_path = os.path.join(path, img)

      img_array = cv.imread(img_path)
      gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
      
      faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

      for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        features.append(faces_roi)
        labels.append(label)
create_train()
print("Training complete")

features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the features list and labels list
face_recognizer.train(features, labels)
if not os.path.exists('trained_data'):
    os.makedirs('trained_data')
face_recognizer.save("trained_data/face_trained.yml")
np.save("trained_data/features.npy", features)
np.save("trained_data/labels.npy", labels)
