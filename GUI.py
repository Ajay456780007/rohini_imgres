import cv2
import numpy as np
import pandas as pd
from demo import Segmentation_prior_model
from keras.models import Model, load_model
from tkinter import filedialog, messagebox
import pydicom
from Sub_Functions.Preprocessing import Preprocessing
import tkinter as tk
from PIL import Image
from PIL import Image, ImageTk

root = tk.Tk()
root.geometry("900x2500")


def Load_DB1():
    try:
        model = load_model(f"Saved_model/{DB}/DB1.h5")
    except:
        print("Model loaded Successfully")


def Select_csv_file():
    global DB, file, f_data

    file = filedialog.askopenfilename(filetypes=[("DCM FILES", "*.dcm")], initialdir="Dataset/")

    data = pydicom.dcmread(file)

    f_data = data.pixel_array

    pil = Image.fromarray(f_data)
    pil = pil.resize((250, 250))

    tk_color = ImageTk.PhotoImage(pil)

    label = tk.Label(original_place, image=tk_color)
    label.image = tk_color
    label.pack()


def Preprocessing_1():
    global ROI, f_data, Prep

    ROI_EXt = Preprocessing(f_data)

    Prep = ROI_EXt.denoise_image()

    # rgb = cv2.cvtColor(Prep, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(Prep)
    pil = pil.resize((250, 250))

    tk_color = ImageTk.PhotoImage(pil)

    label = tk.Label(preprocess_place, image=tk_color)
    label.image = tk_color
    label.pack()


def ROI_Extraction_1():
    global Roi_ext

    R = Preprocessing(Prep)

    Roi_ext = R.ROI_Extraction()

    # rgb = cv2.cvtColor(Roi_ext, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(Roi_ext)
    pil = pil.resize((250, 250))

    tk_pre = ImageTk.PhotoImage(pil)

    label = tk.Label(ROI_place, image=tk_pre)
    label.image = tk_pre
    label.pack()


def Seg_model():
    global seg_image, fin_mask

    mask = Segmentation_prior_model(Roi_ext)

    img_256 = cv2.resize(Roi_ext, (256, 256))
    roi11 = cv2.bitwise_and(img_256, img_256, mask=mask[0].astype(np.uint8))
    roi22 = cv2.bitwise_and(img_256, img_256, mask=mask[1].astype(np.uint8))
    roi33 = cv2.bitwise_and(img_256, img_256, mask=mask[2].astype(np.uint8))
    roi144 = cv2.bitwise_and(img_256, img_256, mask=mask[3].astype(np.uint8))
    roi155 = cv2.bitwise_and(img_256, img_256, mask=mask[4].astype(np.uint8))
    roi166 = cv2.bitwise_and(img_256, img_256, mask=mask[5].astype(np.uint8))
    roi177 = cv2.bitwise_and(img_256, img_256, mask=mask[6].astype(np.uint8))
    roi188 = cv2.bitwise_and(img_256, img_256, mask=mask[7].astype(np.uint8))

    std1 = np.std(roi11)
    std2 = np.std(roi22)
    std3 = np.std(roi33)
    std4 = np.std(roi144)
    std5 = np.std(roi155)
    std6 = np.std(roi166)
    std7 = np.std(roi177)
    std8 = np.std(roi188)

    std_f = [std1, std2, std3, std4, std5, std6, std7, std8]
    roi_index = [roi11, roi22, roi33, roi144, roi155, roi166, roi177, roi188]

    max_out = np.max(std_f)
    max_index = None
    for index, i in enumerate(std_f):
        if max_out == i:
            max_index = index

    fin_mask = roi_index[max_index]

    pil = Image.fromarray(fin_mask)
    pil = pil.resize((250, 250))

    tk_color = ImageTk.PhotoImage(pil)

    label = tk.Label(Segmentation_place, image=tk_color)
    label.image = tk_color
    label.pack()


Load_data1 = tk.Button(root, text="Load DB1", command=Load_DB1, font=("Arial", 14))
Load_data1.place(x=20, y=10)

Load_data2 = tk.Button(root, text="Load DB1", command=Load_DB1, font=("Arial", 14))
Load_data1.place(x=100, y=10)

Load_csv_file = tk.Button(root,text="Select File",command=Select_csv_file,font=("Arial",16))
Load_csv_file.place(x=790, y=10)
Preprocessing1 = tk.Button(root, text="Preprocessing", command=Preprocessing_1, font=("Arial", 16))
Preprocessing1.place(x=250, y=10)

Roi_Extraction_button = tk.Button(root, text="ROI_EXTRACTION", command=ROI_Extraction_1, font=("Arial", 16))
Roi_Extraction_button.place(x=400, y=10)

Segmentation_Button = tk.Button(root, text="Segmentation", command=Seg_model, font=("Arial", 16))
Segmentation_Button.place(x=650, y=10)

original_place = tk.Frame(root, width=250, height=250, bg="white", highlightbackground="black", highlightthickness=1)
# original_place.pack_propagate(False)
original_place.place(x=50, y=100)

preprocess_place = tk.Frame(root, width=250, height=250, bg="white", highlightbackground="black", highlightthickness=1)
# preprocess_place.pack_propagate(False)
preprocess_place.place(x=350, y=100)

ROI_place = tk.Frame(root, width=250, height=250, bg="white", highlightbackground="black", highlightthickness=1)
# ROI_place.pack_propagate(False)
ROI_place.place(x=650, y=100)

Segmentation_place = tk.Frame(root, width=250, height=250, bg="white", highlightbackground="black",
                              highlightthickness=1)
# deep_hog_place.pack_propagate(False)
Segmentation_place.place(x=960, y=100)

root.mainloop()
