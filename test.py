# import streamlit as st
# import zipfile
# from PIL import Image
# import io

# def extract_images_from_excel(file):
#     try:
#         # Read Excel file
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             # Extract images from the xl/media directory
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Display sheet names
#             st.write("Extracted Images:")
#             for image_file in image_files:
#                 image_data = zip_ref.read(image_file)
#                 img = Image.open(io.BytesIO(image_data))
#                 st.image(img, caption=image_file, use_column_width=True)

#     except Exception as e:
#         st.error(f"Error: {e}")

# def main():
#     st.title("Excel Image Extractor")

#     # File uploader
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

#     if file is not None:
#         # Extract images from the uploaded Excel file
#         extract_images_from_excel(file)

# if __name__ == "__main__":
#     main()


import cv2
import pytesseract
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)  # Bring the window to the front
    file_path = filedialog.askopenfilename(title='Select Image File', filetypes=[('Image Files', '*.png *.jpg *.jpeg *.bmp')])
    root.destroy()
    return file_path

def extract_table_from_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Preprocess the image if required (e.g., resizing, filtering, etc.)
    # You may need to experiment with preprocessing based on the quality of your images

    # Perform OCR
    text = pytesseract.image_to_string(image)

    # Parse extracted text to identify the table structure
    # This step heavily depends on the format and structure of the table in the image
    # You may need to use regex or other techniques to extract tabular data

    # Example of parsing extracted text (modify according to your table structure)
    table_data = [line.split('\t') for line in text.strip().split('\n')]

    # Convert to DataFrame
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    return df

def main():
    # Path to Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Nikhil\Downloads\5.3.4 source code\tesseract-ocr-tesseract-bc059ec\src\tesseract.cpp'

    # Allow user to choose image file
    image_path = open_file_dialog()
    if not image_path:
        print("No file selected.")
        return

    # Extract table from image
    try:
        df = extract_table_from_image(image_path)
    except Exception as e:
        print(f"Error extracting table from image: {e}")
        return

    # Export DataFrame to Excel or display it
    print("\nExtracted Table:")
    print(df)

if __name__ == "__main__":
    main()
