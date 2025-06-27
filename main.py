import cv2
import time
import numpy as np
from hand_gesture import HandGesture
from utils import save_image

cap = cv2.VideoCapture(0)
detector = HandGesture()

zoom_level = 1.0
timer_started = False
timer_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, hand_landmarks = detector.find_hands(frame)

    if hand_landmarks:
        landmarks = hand_landmarks[0]

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        palm_open = all(pt[1] < landmarks[0][1] for pt in finger_tips)

        if palm_open:
            save_image(frame)
            time.sleep(1)

        dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
        zoom_level = np.interp(dist, [30, 150], [0.8, 1.5])

        fingers_up = sum(1 for pt in [index_tip, middle_tip, ring_tip] if pt[1] < landmarks[0][1])

        if fingers_up == 3 and not timer_started:
            timer_started = True
            timer_start_time = time.time()

    h, w, _ = frame.shape
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, 0, zoom_level)
    frame = cv2.warpAffine(frame, M, (w, h))

    if timer_started:
        elapsed = time.time() - timer_start_time
        cv2.putText(frame, f'Timer: {3 - int(elapsed)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        
        if elapsed >= 3:
            save_image(frame)
            timer_started = False

    cv2.imshow("Hand Gesture Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
