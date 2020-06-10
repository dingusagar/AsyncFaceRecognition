from ISR.models import RRDN, RDN
import cv2

class ImageEnhancement:
    def __init__(self, method):
        self.method = method
        print('Initialising Super Resolution model of type : {}'.format(method))
        
        if method == None:
            print('No Super Resolution models initalised for method : {}'.format(method))
            self.model = None
        elif method == 'gans':
            self.model = RRDN(weights='gans')
        elif method == 'psnr-small':
            self.model = RDN(weights='psnr-small')
        elif method == 'psnr-large':
            self.model = RDN(weights='psnr-large')
        else:
            raise Exception('Method :{} not valid for ImageEnhancement'.format(method))


    def improve_quality(self,image):
        if self.model is None:
            return image
            
        result =  self.model.predict(image)
        result = cv2.resize(result,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_CUBIC)
        return result





