import cv2
import mediapipe as mp


img_path = "images.jpeg"

img  = cv2.imread(img_path)
H, W, _ = img.shape

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(min_detection_confidence = 0.6, model_selection = 0 ) as face_detection:
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	out = face_detection.process(img_rgb)
	# print(out.detections)


	if out.detections is not None:	
		for detection in out.detections:
			location_data = detection.location_data
			bbox = location_data.relative_bounding_box

			x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
			x1 = int(x1 * W)
			y1 = int(y1 * H)
			w = int(w * W)
			h = int(h * H)
			# img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,0,0), 10)
			img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))
			name = input("Enter filename to save (without extension): ")
	

	cv2.imshow(f"{name}.jpg", img)
	cv2.imwrite(f"{name}_blurred.jpg", img)
	cv2.waitKey(0)