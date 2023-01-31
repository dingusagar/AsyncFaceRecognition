# AsyncFaceRecognition with Super Resolution for biometric attendance system 
A demo python application for face recognition based attendance system using enhancement of low resolution images by super resolution deep learning models. 

This was part of a research project on making face recognition models work in low resolutions. [The final project report](https://drive.google.com/file/d/1INO6TiBOROK-CwntE2ZEtOKYeDC_2jr_/view?usp=sharing) with all the details of the research can be found here. 

This demo project was extended from the awesome [pyimagesearch blog for face recognition](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/). Check the blog for good explanations of the basic code.

Uses `face_recognition` and `ISR` libraries for face recognition and Super resolution respectively.

# Install Dependencies


 1. Install dlib library by following this [guide](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)
 2. Install all other packages using the command `pip install -r requirements.txt`
 
# How to Run
After installing the prerequesites, follow the steps

 1. Create a folder `mydatabase` which has several folders each having photos of a single person and the folder name being the identity (name) of that person. Check the sample `dataset3` folder in the repository.
 2.  Create high level embeddings from these images using the deep learning model and store it in pickle file. To do this just run ` python encode_faces.py -e my_encodings.pickle -i mydatabase -d hog` . This will create a file called `my_encodings.pickle`. Details about other options are explained in the python script.
 3. Run the application with encoding file as an argument. `python recognize_faces_video_async.py -e  my_encodings.pickle  -d hog`
 4. Once the application starts, face detection happens in real time. Face recognition with resolution enhacement is done on a seperate thread. To start recognising, press the r key on keyboard. To store the attendance of the recognised faces on a csv file, press s. To quit the program, press q.

# Demo Video

[![Alt text](https://img.youtube.com/vi/Z0n_G-kh5n0/0.jpg)](https://www.youtube.com/watch?v=Z0n_G-kh5n0)

# Change Super Resolution Models
In order to try out various super resolution models for enhancing the resolution before recognition task, 
change the value of `IMAGE_SUPER_RESOLUTION_METHOD` in the file `recognize_faces_video_async.py`

    IMAGE_SUPER_RESOLUTION_METHOD = 'psnr-small'
    IMAGE_SUPER_RESOLUTION_METHOD = 'psnr-large'
    IMAGE_SUPER_RESOLUTION_METHOD = 'gans' # GAN based Super Resolution  
    IMAGE_SUPER_RESOLUTION_METHOD = None # No super resolution
Refer the [docs](https://idealo.github.io/image-super-resolution/) of `ISR` module for more details about the super resolution models. 

