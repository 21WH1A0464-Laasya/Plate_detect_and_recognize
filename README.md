***License-Plate-Recognition***

**Machine learning**

Recognition of the license number plate of vehicles is significant for the control and surveillance systems. It's a hard and intensive job for humans to manually recognize all the vehicle number plates. Hence, we built an efficient license number plate recognition system using deep learning that is extended to an application of E-challan.

Software Components:

Software : VS Code for coding, Kaggle for training.

Language : Python

Tools and Libraries:

- Python==3.6
- Keras==2.3.1
- Tensorflow==1.14.0
- Numpy==1.17.4
- Matplotlib==3.2.1
- OpenCV==4.1.0
- sklearn==0.21.3

Model used for Plate Detection : Wpod-net (Warped Planar Object Detection Network) architecture (pre-trained model)

Model used for Character Recognition : MobileNetv2 architecture

Inputs:

Classes: 36 (A-Z, 0-9)

Dataset: 37623 images

Training-Validation images ratio: 9:1

Epochs: 30

Database: 100 records

Operation:

Implemented a wpod-net architecture pre-trained model to detect a license plate from the image of a car.

Trained a mobilenetv2 architecture model which contains CNN to recognize one character from an image which can detect the characters with an accuracy of 98%.

Used this trained model in the project to detect all the characters from the number plate image.

Created a sample SQL database of 100 records to store the details of vehicle owners like owner name, email, and car number.

Used the previously detected characters and the database to send mails to the vehicle owners regarding the violation of rules using the smtplib module and python language.

Results:

Training accuracy: 98.03%

Training loss: 5.86%

Validation accuracy: 97.34%

Validation loss: 8.69%

References:

Montazzolli, Sérgio & Jung, Claudio. (2018). License Plate Detection and Recognition in Unconstrained Scenarios. ( Link: https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf )

https://techvidvan.com/tutorials/python-project-license-number-plate-recognition/

Dataset and wpod-net pre trained model: https://github.com/quangnhat185/Plate_detect_and_recognize

https://realpython.com/python-send-email/
