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
    # Root DCM files path
    root = "Dataset/DB1/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM"
    # CSV data contains file paths in it
    metadata_csv = "Dataset/DB1/manifest-ZkhPvrLo5216730872708713142/metadata.csv"
    # csv file containing the class names
    calc_test = "Dataset/DB1/manifest-ZkhPvrLo5216730872708713142/calc_case_description_test_set.csv"
    # reading the metadata file
    metadata = pd.read_csv(metadata_csv)
    # reading the labels file
    label_df = pd.read_csv(calc_test)
    # creating a map
    pathology_map = {
        "BENIGN_WITHOUT_CALLBACK": 0,
        "BENIGN": 1,
        "MALIGNANT": 2
    }
    # empty list to store the final data
    images = []
    masks = []
    labels = []
    # to find the matching labels from csv file
    cc_rows = metadata[
        metadata["Subject ID"].astype(str).str.contains("_LEFT_CC$", regex=True)
    ]
    # total count of files
    total_files = len(cc_rows)
    # iterating over files
    for idx, (_, row) in enumerate(cc_rows.iterrows()):

        cprint(f"Reading file {idx + 1}/{total_files}....", color="magenta", on_color="on_grey", attrs=["bold"])
        subject_id = row["Subject ID"]
        # combining the folder name with base to get the full path of the image folder
        original_folder = os.path.join(root, subject_id)
        # combining the folder name with base to get the full path of the mask folder
        mask_folder = os.path.join(root, subject_id + "_1")
        # if that folder not exist slip it
        if not os.path.exists(original_folder):
            continue
        # if that folder not exist slip it
        if not os.path.exists(mask_folder):
            continue
        # getting the patient id
        patient_id = re.search(r"P_\d+", subject_id).group()
        # getting the matching labels
        label_row = label_df[
            (label_df["patient_id"] == patient_id) &
            (label_df["left or right breast"] == "LEFT") &
            (label_df["image view"] == "CC")
            ]

        if len(label_row) == 0:
            continue

        pathology = label_row.iloc[0]["pathology"]

        if pathology not in pathology_map:
            continue

        original_dcm = None
        # finalizing the dcm file
        for r, d, f in os.walk(original_folder):
            for file in f:
                if file.endswith(".dcm"):
                    original_dcm = os.path.join(r, file)
                    break

        mask_files = []
        # finalizing the mask file
        for r, d, f in os.walk(mask_folder):
            for file in f:
                if file.endswith(".dcm"):
                    mask_files.append(os.path.join(r, file))

        mask_files = sorted(mask_files)

        if original_dcm is None:
            continue

        if len(mask_files) < 2:
            continue

        # try:
        # loading the dcm file
        img = pydicom.dcmread(original_dcm).pixel_array
        # loading the mask file
        mask = pydicom.dcmread(mask_files[1]).pixel_array

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
        os.makedirs(f"Image_Results/{DB}/Sample{idx + 1}/", exist_ok=True)
        plt.savefig(f"Image_Results/{DB}/Sample{idx + 1}/Original.jpg", bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        plt.imshow(ROI_EXT2, cmap="gray")
        plt.axis("off")
        os.makedirs(f"Image_Results/{DB}/Sample{idx + 1}/", exist_ok=True)
        plt.savefig(f"Image_Results/{DB}/Sample{idx + 1}/Preprocessing.jpg", bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        plt.imshow(ROI_EXT, cmap="gray")
        plt.axis("off")
        os.makedirs(f"Image_Results/{DB}/Sample{idx + 1}/", exist_ok=True)
        plt.savefig(f"Image_Results/{DB}/Sample{idx + 1}/ROI_Ext.jpg", bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        plt.imshow(Segmented_image, cmap="gray")
        plt.axis("off")
        os.makedirs(f"Image_Results/{DB}/Sample{idx + 1}/", exist_ok=True)
        plt.savefig(f"Image_Results/{DB}/Sample{idx + 1}/Segmented_image.jpg", bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        plt.imshow(HCMBSP, cmap="gray")
        plt.axis("off")
        os.makedirs(f"Image_Results/{DB}/Sample{idx + 1}/", exist_ok=True)
        plt.savefig(f"Image_Results/{DB}/Sample{idx + 1}/Segmented_image.jpg", bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        plt.imshow(Resnet151, cmap="gray")
        plt.axis("off")
        os.makedirs(f"Image_Results/{DB}/Sample{idx + 1}/", exist_ok=True)
        plt.savefig(f"Image_Results/{DB}/Sample{idx + 1}/Segmented_image.jpg", bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        os.makedirs(f"Image_Results/DB/Sample{idx + 1}/GLCM_Feat/", exist_ok=True)
        plt.imshow(GLCM_Features[:, :, 0], cmap="gray")
        plt.savefig(f"Image_Results/DB/Sample{idx + 1}/GLCM_Feat/Energy.jpg", dpi=800)
        plt.close()
        plt.imshow(GLCM_Features[:, :, 1], cmap="gray")
        plt.savefig(f"Image_Results/DB/Sample{idx + 1}/GLCM_Feat/disimilarity.jpg", dpi=800)
        plt.close()
        plt.imshow(GLCM_Features[:, :, 2], cmap="gray")
        plt.savefig(f"Image_Results/DB/Sample{idx + 1}/GLCM_Feat/Homogenity.jpg", dpi=800)
        plt.close()
        plt.imshow(GLCM_Features[:, :, 3], cmap="gray")
        plt.savefig(f"Image_Results/DB/Sample{idx + 1}/GLCM_Feat/Entropy.jpg", dpi=800)
        plt.close()
        plt.imshow(GLCM_Features[:, :, 4], cmap="gray")
        plt.savefig(f"Image_Results/DB/Sample{idx + 1}/GLCM_Feat/Contrast.jpg", dpi=800)
        plt.close()

        # pathology_map = {
        #     "BENIGN_WITHOUT_CALLBACK": 0,
        #     "BENIGN": 1,
        #     "MALIGNANT": 2
        # }

        if pathology == "BENIGN":
            grade ="- Grade-1"
        elif pathology == "MALIGNANT":
            grade = "- Grade-2"
        else:
            grade = ""
        # img = np.transpose(np.squeeze(img_np), (1, 2, 0))
        # img = np.ascontiguousarray(img)
        img = np.squeeze(img_np)
        out = write_text_fit(img, class_name=pathology+f"{grade}")
        plt.imshow(out)
        plt.axis("off")
        os.makedirs(f"Image_Results/{DB}/Sample{idx + 1}/", exist_ok=True)
        plt.savefig(f"Image_Results/{DB}/Sample{idx + 1}/Output.jpg", bbox_inches='tight',
                    pad_inches=0)
        plt.close()


Image_Results1()
