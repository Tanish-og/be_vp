import cv2
from detector import detect

cap = cv2.VideoCapture("parking.mp4")   # Put parking.mp4 in Backend/

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = detect(frame)

    annotated = results[0].plot()

    cv2.imshow("Parking Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()