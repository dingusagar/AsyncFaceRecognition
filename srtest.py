# Sample program to test the Super Resolution Feature that improves the resolution of images based on deep learning models
# Change the input image path according to your needs

from enhanced_recognition import ImageEnhancement
from face_recognition import load_image_file
from matplotlib import pyplot as plt
import cv2
enhancement = ImageEnhancement()

img = cv2.imread('saved_frame.png')
transformed = enhancement.improve_quality(img,type='gans')
transformed = cv2.resize(transformed,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC)
cv2.imwrite('transformed.png',transformed)
