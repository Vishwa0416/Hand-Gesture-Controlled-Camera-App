import mediapipe as mp
import cv2

class HandGesture:
    def __init__(self, max_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        hand_landmarks = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
                landmarks = []
                h, w, _= img.shape
                for lm in handLms.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
                hand_landmarks.append(landmarks)

        return img, hand_landmarks

    def count_fingers(self, landmarks):
        """
        Counts visible fingers based on landmark positions.
        Assumes landmarks is a list of (x, y) for 21 hand landmarks.
        Returns: number of fingers up
        """
        if not landmarks or len(landmarks) != 21:
            return 0

        fingers = []

        if landmarks[4][0] > landmarks[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        for tip in [8, 12, 16, 20]:
            if landmarks[tip][1] < landmarks[tip - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)
