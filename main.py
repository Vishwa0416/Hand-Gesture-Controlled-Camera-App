import cv2
import time
from hand_gesture import HandGesture
import os

# Create the folder if it doesn't exist
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = HandGesture()

last_trigger_time = 0
cooldown = 3  # seconds to wait before allowing next capture

while True:
    success, img = cap.read()
    img, hands = detector.find_hands(img)

    if hands:
        landmarks = hands[0]
        fingers_up = detector.count_fingers(landmarks)

        current_time = time.time()

        if fingers_up > 0:
            timer = fingers_up  # Timer based on number of visible fingers

            if current_time - last_trigger_time > cooldown:
                print(f"Taking photo in {timer} second(s)...")
                time.sleep(timer)
                
                # Construct save path inside 'captured_images' folder
                filename = os.path.join(save_dir, f'photo_{int(time.time())}.jpg')
                cv2.imwrite(filename, img)
                print(f"Photo saved to {filename}")
                
                last_trigger_time = time.time()

    cv2.imshow("Hand Gesture Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
