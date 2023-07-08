import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog  # filedialog for uploading images
from tkinter import PhotoImage 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
from sklearn.preprocessing import LabelEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import mail  # user defined python file
import sqlite3

window = tk.Tk()
w = 900
h = 700  
ws = window.winfo_screenwidth() 
hs = window.winfo_screenheight()  
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
window.geometry('%dx%d+%d+%d' % (w, h, x, y))  # size of the window
window.title('License Number PlateRecognition')


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor


# Displaying a button to browse the vehicle image
def show_upload_button(welcome_btn):
    welcome_btn.pack_forget()
    # creating button to upload image, upload_image() is called when user clicks "Browse image" button
    upload = Button(window, text="Browse image", command=upload_image, padx=2, pady=2)
    # padx, pady - for space between border and text - padding
    upload.configure(background='#57373d', foreground='white', font=('times new roman', 15, 'bold'))
    upload.pack()
    upload.place(x=210, y=550)

    # displaying the input image
    input_image.pack()
    input_image.place(x=40, y=200)

    # displaying the final output text
    label.pack()
    label.place(x=70, y=600)


# Browse image - displaying filedialog
def upload_image():
    try:
        file_path = filedialog.askopenfilename()  # browse the location of image
        uploaded = Image.open(file_path)  # open the image in the specified location
        # resizing the image to fit in the window
        uploaded.thumbnail(((window.winfo_width() / 2.5), (window.winfo_height() / 2.5)))
        inp_img = ImageTk.PhotoImage(uploaded)
        input_image.configure(image=inp_img)
        input_image.image = inp_img
        label.configure(text='')
        show_crop_button(file_path)
    except:
        pass


# Pre-process each character image and predict the output label using trained model
def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction


# creating a connection with the vehicle database
def get_connection():
    connection = sqlite3.connect('vehicledatabase.db')
    return connection


def show_crop_button(file_path):
    crop_b = Button(window, text="Detect License Plate", command=lambda: detect_and_recognize(file_path), padx=2, pady=2)
    crop_b.configure(background='#57373d', foreground='white', font=('times new roman', 15, 'bold'))
    crop_b.place(x=600, y=550)


# Input - vehicle image location
# Output - license plate detect, recognize, send mail
def detect_and_recognize(file_path):
    vehicle, LpImg, _ = get_plate(file_path)
    if len(LpImg):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Applied inversed thresh_binary
        binary = cv2.threshold(blur, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Dilation
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    else:
        print("No license plate detected")

    # visualize results
    fig = plt.figure(figsize=(12, 7))
    plt.rcParams.update({"font.size": 18})
    grid = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    plot_image = [plate_image, gray, blur, binary, thre_mor]
    plot_name = ["plate_image", "gray", "blur", "binary", "dilation"]

    for i in range(len(plot_image)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.title(plot_name[i])
        if i == 0:
            plt.imshow(plot_image[i])
        else:
            plt.imshow(plot_image[i], cmap="gray")
    # plt.show()
    plt.savefig("preprocessing of number plate.png", dpi=300)

    cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # create a copy version "test_roi" of plate_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append character image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 0.5 <= ratio <= 4:  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.4:  # Select contour which has the height larger than 40% of the plate
                # Draw bounding box around digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Separate each character image and append to crop_characters list
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detected {} letters...".format(len(crop_characters)))
    fig = plt.figure(figsize=(10, 6))
    plt.axis(False)
    plt.imshow(test_roi)
    # plt.show()
    plt.savefig('grab_digit_contour.png', dpi=300)

    fig = plt.figure(figsize=(10, 5))
    grid = gridspec.GridSpec(ncols=len(crop_characters), nrows=1, figure=fig)

    for i in range(len(crop_characters)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.imshow(crop_characters[i], cmap="gray")
    # plt.show()
    plt.savefig("segmented_letter.png", dpi=300)

    # Load model architecture, weight and labels
    json_file = open('MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("License_character_recognition_weight.h5")
    print("[INFO] Model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load('license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")

    fig = plt.figure(figsize=(15, 3))
    cols = len(crop_characters)
    grid = gridspec.GridSpec(ncols=cols, nrows=1, figure=fig)

    final_string = ''
    for i, character in enumerate(crop_characters):
        fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character, model, labels))
        plt.title('{}'.format(title.strip("'[]"), fontsize=20))
        final_string += title.strip("'[]")
        plt.axis(False)
        plt.imshow(character, cmap='gray')

    print(final_string)
    # plt.show()
    plt.savefig('final_result.png', dpi=300)

    with open("car number text.txt", "a") as op_file:
        op_file.write(final_string + "\n")

    # to print the detected text
    label.configure(foreground='#20262E', text="Car number: " + final_string)

    # uploading the final result image on GUI window, which is stored in plate_image
    uploaded = Image.open("final_result.png")
    uploaded.thumbnail(((window.winfo_width() / 2.25), (window.winfo_height() / 2.25)))
    plt_img = ImageTk.PhotoImage(uploaded)
    plate_image = Label(window, bd=10)
    plate_image.configure(image=plt_img)
    plate_image.image = plt_img
    plate_image.pack()
    plate_image.place(x=500, y=320)


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


    # set icon for the window
window.iconphoto(True, PhotoImage(file="echallan logo.png"))
loc_icon = Image.open("echallan logo.png")  # display icon on window
loc_icon.thumbnail((360, 240))  # resizing the icon
logo_image = ImageTk.PhotoImage(loc_icon)  # load the image into logo_image
window.configure(background='#EEEEEE')  # setting a background colour

    # creating label for OCR detected text
label = Label(window, background='#EEEEEE', font=('times new roman', 30, 'bold'))

    # creating labels for cropped plate image and input car image
input_image = Label(window, bd=10)

    # creating label for displaying roll numbers on gui
label1 = Label(window, background='#EEEEEE', font=('times new roman', 15, 'bold'))
label1.configure(foreground='#000000', text="Jahnavi Muppalla\nLaasya Modugamudi")
label1.pack()
label1.place(x=680, y=40)

heading = Label(window, image=logo_image)  # creating a label of the logo image
heading.configure(background='#EEEEEE')  # setting same background colour as that of gui
heading.pack()  # adding the heading icon to gui

    # Creating a photoimage object to use image
photo = PhotoImage(file="welcome pic.png")
welcome_btn = Button(window, image=photo)# set image on button
welcome_btn.configure(command=lambda: show_upload_button(welcome_btn))
welcome_btn.configure(borderwidth=10)
welcome_btn.pack(side=BOTTOM)


window.mainloop()

cv2.destroyAllWindows()
