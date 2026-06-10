import numpy as np
import pandas as pd
import os
import re
import pydicom
import xml.etree.ElementTree as ET
import plistlib
from termcolor import cprint
from Sub_Functions.Preprocessing import Preprocessing
from Sub_Functions.Features import Features_1
from deep_segmentation_prior.source.foreground_background._utils import *
from demo import Segmentation_prior_model
import matplotlib.pyplot as plt
import cv2

DB = "CBIS_DDSM"


def write_text_fit(frame_with_text, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    text_color = (0, 0, 0)  # Black text
    background_color = (255, 255, 255)  # White background

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        class_name, font, font_scale, thickness
    )

    # Center text at top
    x = (frame_with_text.shape[1] - text_width) // 2
    y = text_height + 10

    # Draw white rectangle background
    cv2.rectangle(
        frame_with_text,
        (x - 8, y - text_height - 8),
        (x + text_width + 8, y + 8),
        background_color,
        -1
    )

    # Put black text on white background
    cv2.putText(
        frame_with_text,
        class_name,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    frame_with_text = cv2.cvtColor(frame_with_text, cv2.COLOR_BGR2RGB)

    return frame_with_text


def Image_Results1():
    # initializing empty list for storing the images
    images = []
    labels = []
    # base path of the folder containing dcm files
    base_path = "Dataset/DB2/inbreast/ALL-IMGS"
    # listing files inside the base path
    listed = os.listdir(base_path)

    temp_checker1 = None
    for file in range(len(listed) - 1):
        cprint(f"Reading file {file + 1}/{len(listed)}....", color="magenta", on_color="on_grey", attrs=["bold"])
        # getting the id of each files
        splitted = listed[file].split("_")
        # the name is at index 1 of the filename
        temp_checker = splitted[1]
        # the id is at the index 0 of the filename
        id = splitted[0]
        # checking to avoid same id again
        if temp_checker != temp_checker1:
            # getting the complete dcm file path
            complete_path = "Dataset/DB2/inbreast/ALL-IMGS/" + listed[file]

            try:
                # getting the matching xml path
                xml_path = "Dataset/DB2/AllXML/" + id + ".xml"
                # class mapping to convert the labels to integer
                class_mapping = {'Normal': 0, 'Mass': 1, 'Micros': 2, 'Distortion': 3, 'Asymmetry': 4}
                # opening the xml fie and finding the classes
                with open(xml_path, 'rb') as f:
                    data = plistlib.load(f)

                lesion_name = data['Images'][0]['ROIs'][0]['Name']

                # mapping the class id
                class_id = class_mapping.get(lesion_name, 0)
                # reading the dicom images
                img = pydicom.dcmread(complete_path).pixel_array

                img_float = img.astype(float)
                img_min = np.min(img_float)
                img_max = np.max(img_float)

                if img_max - img_min == 0:
                    # Avoid zero-division errors if the image is completely blank
                    normalized_img = np.zeros_like(img_float)
                else:
                    normalized_img = (img_float - img_min) / (img_max - img_min)

                img_8bit = (normalized_img * 255.0).astype(np.uint8)
                pil_image = Image.fromarray(img_8bit)
                pil_image = pil_image.resize((256, 256))
                # img_pil = Image.open(pil_image)
                # img_pil = crop_image_by_multiplier(pil_image, d=32)

                img_np = pil_to_np(pil_image)

                # Instance created for preprocessing
                Pre = Preprocessing(img)
                # Extracting ROI
                ROI_EXT = Pre.ROI_Extraction()

                Pre2 = Preprocessing(ROI_EXT)

                ROI_EXT2 = Pre2.denoise_image()

                Segmented_image = Segmentation_prior_model(ROI_EXT)

                Feat = Features_1(Segmented_image)
                GLCM_Features = Feat.GLCM_Features_1()
                HCMBSP = Feat.Hybrid_colormap_based_Strut_pattern()
                Resnet151 = Feat.Resnet151()

                plt.imshow(np.transpose(img_np, (1, 2, 0)), cmap="gray")
                plt.axis("off")
                os.makedirs(f"Image_Results/{DB}/Sample{file + 1}/", exist_ok=True)
                plt.savefig(f"Image_Results/{DB}/Sample{file + 1}/Original.jpg", bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(ROI_EXT2, cmap="gray")
                plt.axis("off")
                os.makedirs(f"Image_Results/{DB}/Sample{file + 1}/", exist_ok=True)
                plt.savefig(f"Image_Results/{DB}/Sample{file + 1}/Preprocessing.jpg", bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(ROI_EXT, cmap="gray")
                plt.axis("off")
                os.makedirs(f"Image_Results/{DB}/Sample{file + 1}/", exist_ok=True)
                plt.savefig(f"Image_Results/{DB}/Sample{file + 1}/ROI_Ext.jpg", bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(Segmented_image, cmap="gray")
                plt.axis("off")
                os.makedirs(f"Image_Results/{DB}/Sample{file + 1}/", exist_ok=True)
                plt.savefig(f"Image_Results/{DB}/Sample{file + 1}/Segmented_image.jpg", bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(HCMBSP, cmap="gray")
                plt.axis("off")
                os.makedirs(f"Image_Results/{DB}/Sample{file + 1}/", exist_ok=True)
                plt.savefig(f"Image_Results/{DB}/Sample{file + 1}/Segmented_image.jpg", bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                plt.imshow(Resnet151, cmap="gray")
                plt.axis("off")
                os.makedirs(f"Image_Results/{DB}/Sample{file + 1}/", exist_ok=True)
                plt.savefig(f"Image_Results/{DB}/Sample{file + 1}/Segmented_image.jpg", bbox_inches='tight',
                            pad_inches=0)
                plt.close()

                os.makedirs(f"Image_Results/DB/Sample{file + 1}/GLCM_Feat/", exist_ok=True)
                plt.imshow(GLCM_Features[:, :, 0], cmap="gray")
                plt.savefig(f"Image_Results/DB/Sample{file + 1}/GLCM_Feat/Energy.jpg", dpi=800)
                plt.close()
                plt.imshow(GLCM_Features[:, :, 1], cmap="gray")
                plt.savefig(f"Image_Results/DB/Sample{file + 1}/GLCM_Feat/disimilarity.jpg", dpi=800)
                plt.close()
                plt.imshow(GLCM_Features[:, :, 2], cmap="gray")
                plt.savefig(f"Image_Results/DB/Sample{file + 1}/GLCM_Feat/Homogenity.jpg", dpi=800)
                plt.close()
                plt.imshow(GLCM_Features[:, :, 3], cmap="gray")
                plt.savefig(f"Image_Results/DB/Sample{file + 1}/GLCM_Feat/Entropy.jpg", dpi=800)
                plt.close()
                plt.imshow(GLCM_Features[:, :, 4], cmap="gray")
                plt.savefig(f"Image_Results/DB/Sample{file + 1}/GLCM_Feat/Contrast.jpg", dpi=800)
                plt.close()

                # pathology_map = {
                #     "BENIGN_WITHOUT_CALLBACK": 0,
                #     "BENIGN": 1,
                #     "MALIGNANT": 2
                # }

                class_mapping = {'Normal': 0, 'Mass': 1, 'Micros': 2, 'Distortion': 3, 'Asymmetry': 4}

                if lesion_name == "Asymmetry":
                    grade ="- Grade-1"
                elif lesion_name == "Micros":
                    grade = "- Grade-2"
                elif lesion_name == "Mass":
                    grade = "- Grade-3"

                elif lesion_name == "Distortion":
                    grade = "- Grade-4"
                else:
                    grade = ""
                # img = np.transpose(np.squeeze(img_np), (1, 2, 0))
                # img = np.ascontiguousarray(img)
                img = np.squeeze(img_np)
                out = write_text_fit(img, class_name=lesion_name+f"{grade}")
                plt.imshow(out)
                plt.axis("off")
                os.makedirs(f"Image_Results/{DB}/Sample{file + 1}/", exist_ok=True)
                plt.savefig(f"Image_Results/{DB}/Sample{file + 1}/Output.jpg", bbox_inches='tight',
                            pad_inches=0)
                plt.close()
            except:
                continue

Image_Results1()
