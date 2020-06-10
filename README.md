# AsyncFaceRecognition
A demo python application for face recognition based attendance system using enhancement of low resolution images by super resolution deep learning models. 


# How to Run
After installing the prerequesites, follow the steps

 1. Create a folder `mydatabase` which has several folders each having photos of a single person and the folder name being the identity (name) of that person. Check the sample `dataset3` folder in the repository.
 2.  Create high level embeddings from these images using the deep learning model and store it in pickle file. To do this just run ` python encode_faces.py -e my_encodings.pickle -i mydatabase -d hog` . This will create a file called `my_encodings.pickle`. Details about other options are explained in the python script.
 3. Run the application with encoding file as an argument. `python recognize_faces_video_async.py -e  my_encodings.pickle  -d hog`
 4. Once the application starts, face detection happens in real time. Face recognition with resolution enhacement is done on a seperate thread. To start recognising, press the r key on keyboard. To store the attendance of the recognised faces on a csv file, press s. To quit the program, press q.
