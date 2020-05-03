from ISR.models import RRDN, RDN
import cv2

class ImageEnhancement:
    def __init__(self):
        self.ganModel = RRDN(weights='gans')
        

    def improve_quality(self,image,type):
        if type == 'gans':
            result =  self.ganModel.predict(image)
            result = cv2.resize(result,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_CUBIC)
            return result



