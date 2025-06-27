import cv2
import os
from datetime import datetime

def save_image(img):
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')

    filename = f'captured_images/photo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
    cv2.imwrite(filename, img)
    print(f'Image saved: {filename}')