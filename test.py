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
