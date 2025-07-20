import cv2
import numpy as np

from PIL import Image
cap = cv2.VideoCapture(0)


def get_limits(color):
    c = np.uint8([[color]]) 
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  

    if hue >= 165:  
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15: 
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

colours = {
    'white':    [255, 255, 255],
    'black':    [0, 0, 0],
    'red':      [0, 0, 255],
    'green':    [0, 255, 0],
    'blue':     [255, 0, 0],
    'yellow':   [0, 255, 255],
    'cyan':     [255, 255, 0],
    'magenta':  [255, 0, 255],
    'orange':   [0, 165, 255],
    'purple':   [128, 0, 128],
    'pink':     [203, 192, 255],
    'brown':    [42, 42, 165],
    'gray':     [128, 128, 128],
    'indigo':   [130, 0, 75],
    'maroon':   [0, 0, 128],
    'navy':     [128, 0, 0],
    'gold':     [0, 215, 255]
}

colour_chosen = input("what colour do you want to detect ").lower()
colour = colours[colour_chosen]


while True:
	ret, frame = cap.read()

	hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lowerLimit, upperLimit = get_limits(color=colour)

	mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

	mask_ = Image.fromarray(mask)

	bbox = mask_.getbbox()
	if bbox is not None:
		x1, y1, x2, y2 = bbox
		if colour_chosen == "green":
			frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)
		else:
			frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

cv2.destroyAllWindows()
