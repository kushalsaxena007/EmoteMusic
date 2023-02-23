import cv2
from deepface import DeepFace
import webbrowser

count = 0

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    count = count + 1
    ret, frame = cap.read()
    result = DeepFace.analyze(frame, actions = ['emotion'])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 3, (0, 0, 225), 2, cv2.LINE_4)
    
    if count == 50:
        emotion = result['dominant_emotion']
        break

    cv2.imshow('Original video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(emotion)

if emotion == 'sad':
    webbrowser.open('https://www.youtube.com/watch?v=kbKkbNY8Vnw')
elif emotion == 'happy':
    webbrowser.open('https://www.youtube.com/watch?v=dQS5-X5plpc')
elif emotion == 'angry':
    webbrowser.open('https://www.youtube.com/watch?v=8CEJoCr_9UI')
elif emotion == 'neutral':
    webbrowser.open('https://www.youtube.com/watch?v=uuF45i8bOF0')