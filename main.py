import cv2
import time
import math
from hand_gesture import HandGesture
import os

save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = HandGesture()

countdown_started = False
start_timer_time = 0
timer_duration = 0
last_trigger_time = 0
cooldown = 3

while True:
    success, img = cap.read()
    img, hands = detector.find_hands(img)

    current_time = time.time()
    fingers_up = 0

    if hands:
        landmarks = hands[0]
        fingers_up = detector.count_fingers(landmarks)

        # Start countdown only for 1 to 5 fingers, no other condition
        if fingers_up in [1, 2, 3, 4, 5] and not countdown_started and (current_time - last_trigger_time) > cooldown:
            timer_duration = fingers_up
            start_timer_time = current_time
            countdown_started = True
            print(f"Countdown started for {timer_duration} second(s)...")

    if countdown_started:
        elapsed = current_time - start_timer_time
        remaining = math.ceil(timer_duration - elapsed)

        if remaining > 0:
            cv2.putText(img, f"Taking photo in {remaining}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
        else:
            filename = os.path.join(save_dir, f'photo_{int(time.time())}.jpg')
            cv2.imwrite(filename, img)
            print(f"Photo saved to {filename}")

            last_trigger_time = time.time()
            countdown_started = False

    cv2.imshow("Hand Gesture Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
