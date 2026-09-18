import cv2
from detector import detect

image = cv2.imread("test.jpg")      # Put a parking image in Backend/

results = detect(image)

annotated = results[0].plot()

cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()