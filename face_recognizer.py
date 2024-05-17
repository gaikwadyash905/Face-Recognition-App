import cv2 as cv
import os

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def load_trained_data():
    haar_cascade_path = "haarcascade_frontalface_alt.xml"
    face_recognizer_path = "trained_data/face_trained.yml"
    
    if not os.path.exists(haar_cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found at {haar_cascade_path}")

    if not os.path.exists(face_recognizer_path):
        raise FileNotFoundError(f"Face recognizer file not found at {face_recognizer_path}")

    haar_cascade = cv.CascadeClassifier(haar_cascade_path)
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read(face_recognizer_path)
    
    people = []
    for i in os.listdir(r"training_faces/"):
        people.append(i)
    
    return haar_cascade, face_recognizer, people

def detect_faces_in_image(image_path, haar_cascade, face_recognizer, people, scale=1):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load the image at {image_path}")

    img = rescale_frame(img, scale)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        print(f"Label = {label}, Confidence = {confidence}")
        text = f"{people[label]} ({confidence:.2f})"
        (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 1.0, 2)
        text_x = x
        text_y = y - 10 if y - 10 > text_height else y + text_height + 10
        cv.putText(img, text, (text_x, text_y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Detected Face", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detect_faces_in_webcam(haar_cascade, face_recognizer, people, scale=1):
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = rescale_frame(frame, scale)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(faces_roi)
            print(f"Label = {label}, Confidence = {confidence}")
            text = f"{people[label]} ({confidence:.2f})"
            (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX, 1.0, 2)
            text_x = x
            text_y = y - 10 if y - 10 > text_height else y + text_height + 10
            cv.putText(frame, text, (text_x, text_y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow("Webcam Face Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Load trained data
haar_cascade, face_recognizer, people = load_trained_data()

# Test image path
test_image_path = path\to\test\image.png"

# Detect faces in the given image
# detect_faces_in_image(test_image_path, haar_cascade, face_recognizer, people)

# Webcam for real-time face detection
detect_faces_in_webcam(haar_cascade, face_recognizer, people)
