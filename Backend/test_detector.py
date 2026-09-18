import cv2

from detector import detect
from geometry import get_slot_status

image = cv2.imread("test.jpg")

detections, results = detect(image)

print("Detections:")
print(detections)

status = get_slot_status(detections)

print("\nSlot Status:")
print(status)