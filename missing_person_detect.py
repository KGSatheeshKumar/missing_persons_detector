import cv2
import numpy as np
from PIL import Image
import os

def detect_person():
    # Set up the face recognizer
    path = 'data'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def imgsandlables(path):
        imagePaths = [os.path.join(path, i) for i in os.listdir(path) if i.endswith(('.jpg', '.png'))]
        indfaces = []
        ids = []
        filenames = []
        
        for imagePath in imagePaths:
            img = Image.open(imagePath).convert('L')  # Convert image to grayscale
            imgnp = np.array(img, 'uint8')
            
            filename = os.path.split(imagePath)[-1]
            id_str = filename.split('.')[0]
            id = int(id_str.split('_')[0])
            
            faces = detector.detectMultiScale(imgnp)
            for (x, y, w, h) in faces:
                indfaces.append(imgnp[y:y + h, x:x + w])
                ids.append(id)
                filenames.append(filename)
        
        return indfaces, ids, filenames

    faces, ids, filenames = imgsandlables(path)
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
    else:
        print("No training data found!")
        return None, None, None

    names = ['None']
    max_id = max(ids)
    for i in range(1, max_id + 1):
        names.append(f'User {i}')

    # Initialize webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detected_person = None
    detected_image = None

    while True:
        _, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if (confidence < 100):
                detected_person = names[id]
                detected_image = filenames[ids.index(id)]
                confidence = "  {0}%".format(round(100 - confidence))
                
                # Release camera and close windows after detection
                cam.release()
                cv2.destroyAllWindows()
                return detected_person, confidence, detected_image
            else:
                id = "No Match Found"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) 

        cv2.imshow('Missing Person Detection', img)

        k = cv2.waitKey(10) & 0xFF 
        if k == 27:  # Press 'Esc' to exit
            break

    cam.release()
    cv2.destroyAllWindows()
    return None, None, None

if __name__ == "__main__":
    detect_person() 