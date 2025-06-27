import mediapipe as mp
import cv2

class HandGesture:
    def __init__(self, max_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands)
        self.mp_draw = mp.solutions.drawing_utils

    def find__hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        hand_landmarks = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for lm in handLms.landmark:
                        h, w, _= img.shape
                        landmarks.append((int(lm.x * w), int(lm.y * h)))
                    hand_landmarks.append(landmarks)

        return img, hand_landmarks