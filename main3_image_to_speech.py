import cv2 
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import pyttsx3
engine = pyttsx3.init()

img_path = "quote.jpg"

img = cv2.imread(img_path)

reader = easyocr.Reader(['en'], gpu = False)

text = reader.readtext(img)

paragraph = ''

for t in text:

	bbox, detected_text, score = t
	paragraph += detected_text + ' '



# 	# print(t) 
	
# 	bbox, text, score = t

	# pt1 = tuple(map(int, bbox[0]))  
	# pt2 = tuple(map(int, bbox[2])) 
	# if score > 0.8:
	# 	cv2.rectangle(img, pt1, pt2, (0,255,0), 5)
	# 	cv2.putText(img, text, pt1, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 2)
engine.setProperty('rate', 150)     # slower/faster voice
engine.setProperty('volume', 0.9)   # 0.0 to 1.0

engine.say(paragraph)
engine.runAndWait()
print(paragraph)

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()				