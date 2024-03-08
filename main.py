# import streamlit as st
# import pandas as pd
# from PIL import Image
# import base64
# import google.generativeai as genai


# # Set your API keys for both Google Gemini Vision Pro and Google Gemini
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# # Create an instance of the GenerativeModel class with the model name 'gemini-pro-vision' and set the API key
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# # Create an instance of the GenerativeModel class with the model name 'gemini-pro' and set the API key
# gemini_model = genai.GenerativeModel('gemini-pro')

# # Streamlit app
# def main():
#     st.title("Image To Excel")

#     # Upload image through Streamlit
#     uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_image is not None:
#         # Open the uploaded image using the Pillow library
#         image_pil = Image.open(uploaded_image)

#         # Display the uploaded image
#         st.image(image_pil, caption="Uploaded Image", use_column_width=True)

#         # Prompt for extracting text using Gemini Vision Pro
#         vision_pro_prompt = "Extract text from the provided image."

#         # Generate content using the Google Gemini Vision Pro API to extract text from the image
#         with st.spinner("Extracting Text From The Image..."):
#             vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, image_pil])

#         # Resolve the response to obtain the extracted text
#         vision_pro_response.resolve()

#         # Access the text from parts
#         extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)

#         # Display the extracted text
#         st.subheader("Extracted Text:")
#         st.write(extracted_text)

#         # Prompt for extracting specific information using Gemini Pro
#         gemini_prompt = """
#         Extract specific information from the provided text regarding Basic Terms, including:
#         - Date Announced
#         - Actual Completion Date
#         - Type of Consideration
#         - Consideration Terms
#         - Financing Condition
#         - Jurisdiction
#         - Initial Expected Completion Timeline
#         - Industry
#         - Marketing Period
#         - Go-Shop
#         ignore other information & dont put any special characters like * etc
#         """

#         # Use the Google Gemini API to extract specific information from the text
#         with st.spinner("Extracting Specific Information From The Text ..."):
#             gemini_response = extract_specific_information(extracted_text, gemini_api_key, gemini_prompt)

#         # Display the result from Google Gemini API
#         st.subheader("Extracted Specific Information:")
#         st.write(gemini_response.text)

#         # Create DataFrame and Export to Excel
#         df = create_dataframe(gemini_response.text)

#         # Specify the file path for the Excel file
#         excel_file_path = "output.xlsx"

#         # Increase the column widths
#         col_widths = [max(len(str(value)) + 4, 20) for value in df.iloc[0]]

#         # Create the Excel file with increased column widths
#         with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
#             df.to_excel(writer, index=False, sheet_name='Sheet1')
#             worksheet = writer.sheets['Sheet1']
#             for i, width in enumerate(col_widths):
#                 worksheet.column_dimensions[worksheet.cell(row=1, column=i + 1).column_letter].width = width

#         # Encode the Excel data to base64 for download
#         with open(excel_file_path, 'rb') as file:
#             excel_data = file.read()
#             b64 = base64.b64encode(excel_data).decode()

#         # Create a download link
#         href = f'<a href="data:application/octet-stream;base64,{b64}" download="{excel_file_path}">Click here to download the Excel file</a>'
#         st.markdown(href, unsafe_allow_html=True)

#         # Display the styled DataFrame
#         st.write(df)

# def extract_specific_information(text, api_key, prompt):
#     # Use the Google Gemini API to extract specific information
#     gemini_response = gemini_model.generate_content([prompt, text])
#     gemini_response.resolve()

#     return gemini_response

# def create_dataframe(text):
#     # Parse the extracted text to create a DataFrame
#     lines = text.split("\n")
#     data = {}

#     for line in lines:
#         if ":" in line:
#             key, value = map(str.strip, line.split(":", 1))
#             # Replace invalid characters in column names
#             key = key.replace("=", "").replace("-", "").strip()
#             data[key] = [value]

#     # Create a DataFrame with proper columns and values
#     df = pd.DataFrame(data)
    
#     return df

# if __name__ == "__main__":
#     main()




# import streamlit as st
# import zipfile
# from PIL import Image
# import io
# import pandas as pd
# import base64
# import google.generativeai as genai

# # Set your API keys for both Google Gemini Vision Pro and Google Gemini
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# # Create an instance of the GenerativeModel class with the model name 'gemini-pro-vision' and set the API key
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# # Create an instance of the GenerativeModel class with the model name 'gemini-pro' and set the API key
# gemini_model = genai.GenerativeModel('gemini-pro')

# def extract_images_and_process(file):
#     try:
#         # Read Excel file
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             # Extract images from the xl/media directory
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Display sheet names
#             st.write("Extracted Images and Specific Information:")
            
#             for image_file in image_files:
#                 # Extract sheet name from the image file name
#                 sheet_name = image_file.split('/')[-1].split('.')[0]

#                 # Display sheet name
#                 st.subheader(f"Processing Sheet: {sheet_name}")

#                 # Extract image
#                 image_data = zip_ref.read(image_file)
#                 img = Image.open(io.BytesIO(image_data))
#                 st.image(img, caption=f"Original Image for {sheet_name}", use_column_width=True)

#                 # Prompt for extracting text using Gemini Vision Pro
#                 vision_pro_prompt = "Extract text from the provided image."

#                 # Generate content using the Google Gemini Vision Pro API to extract text from the image
#                 with st.spinner("Extracting Text From The Image..."):
#                     vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])

#                 # Resolve the response to obtain the extracted text
#                 vision_pro_response.resolve()

#                 # Access the text from parts
#                 extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)

#                 # Display the extracted text
#                 st.subheader("Extracted Text:")
#                 st.write(extracted_text)

#                 # Prompt for extracting specific information using Gemini Pro
#                 gemini_prompt = """
#                 Extract specific information from the provided text regarding Basic Terms, including:
#                 - Date Announced
#                 - Actual Completion Date
#                 - Type of Consideration
#                 - Consideration Terms
#                 - Financing Condition
#                 - Jurisdiction
#                 - Initial Expected Completion Timeline
#                 - Industry
#                 - Marketing Period
#                 - Go-Shop
#                 ignore other information & dont put any special characters like * etc
#                 """


#                 # Use the Google Gemini API to extract specific information from the text
#                 with st.spinner("Extracting Specific Information From The Text ..."):
#                     gemini_response = gemini_model.generate_content([gemini_prompt, extracted_text])

#                 # Print the Gemini response for debugging
#                 st.subheader("Gemini Response:")
#                 st.write(gemini_response)

#                 # Resolve the response to obtain the extracted specific information
#                 gemini_response.resolve()

#                 # Access the specific information from parts
#                 extracted_specific_info = gemini_response.candidates[0].content.text

#                 # Display the extracted specific information
#                 st.subheader("Extracted Specific Information:")
#                 st.write(extracted_specific_info)

#                 # Append specific information to the original sheet
#                 append_specific_information_to_sheet(file, sheet_name, extracted_specific_info)

#     except Exception as e:
#         st.error(f"Error: {e}")

# def append_specific_information_to_sheet(excel_path, sheet_name, specific_info):
#     try:
#         # Load the Excel file
#         xls = pd.ExcelFile(excel_path)

#         # Read the existing data from the sheet
#         existing_data = pd.read_excel(xls, sheet_name)

#         # Create a DataFrame with specific information
#         specific_info_df = pd.DataFrame({"Extracted_Specific_Info": [specific_info]})

#         # Concatenate the existing data with specific information
#         updated_data = pd.concat([existing_data, specific_info_df], axis=1)

#         # Save the updated data back to the Excel file
#         with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
#             updated_data.to_excel(writer, index=False, sheet_name=sheet_name)

#     except Exception as e:
#         st.error(f"Error appending specific information to sheet {sheet_name}: {e}")

# def main():
#     st.title("Excel Image and Information Extractor")

#     # File uploader
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

#     if file is not None:
#         # Extract images and process specific information from the uploaded Excel file
#         extract_images_and_process(file)

# if __name__ == "__main__":
#     main()




# import streamlit as st
# import zipfile
# from PIL import Image
# import io
# import pandas as pd
# import base64
# import google.generativeai as genai

# # Set your API keys for both Google Gemini Vision Pro and Google Gemini
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# # Create an instance of the GenerativeModel class with the model name 'gemini-pro-vision' and set the API key
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# # Create an instance of the GenerativeModel class with the model name 'gemini-pro' and set the API key
# gemini_model = genai.GenerativeModel('gemini-pro')

# def extract_images_and_process(file):
#     try:
#         # Read Excel file
#         image_sheet_mapping = {}
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             for image_file in image_files:
#                 sheet_name = image_file.split('/')[-1].split('.')[0]
#                 image_sheet_mapping[image_file] = sheet_name


#                 # Display sheet name
#                 st.subheader(f"Processing Sheet: {sheet_name}")

#                 # Extract image
#                 image_data = zip_ref.read(image_file)
#                 img = Image.open(io.BytesIO(image_data))
#                 st.image(img, caption=f"Original Image for {sheet_name}", use_column_width=True)

#                 # Prompt for extracting text using Gemini Vision Pro
#                 vision_pro_prompt = "Extract text from the provided image."

#                 # Generate content using the Google Gemini Vision Pro API to extract text from the image
#                 with st.spinner("Extracting Text From The Image..."):
#                     vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])

#                 # Resolve the response to obtain the extracted text
#                 vision_pro_response.resolve()

#                 # Access the text from parts
#                 extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)

#                 # Display the extracted text
#                 st.subheader("Extracted Text:")
#                 st.write(extracted_text)

#                 # Prompt for extracting specific information using Gemini Pro
                # gemini_prompt = """
                # Extract specific information from the provided text regarding Basic Terms, including:
                # - Date Announced
                # - Actual Completion Date
                # - Type of Consideration
                # - Consideration Terms
                # - Financing Condition
                # - Jurisdiction
                # - Initial Expected Completion Timeline
                # - Industry
                # - Marketing Period
                # - Go-Shop
                # ignore other information & dont put any special characters like * etc
                # """

#                 # Use the Google Gemini API to extract specific information from the text
#                 with st.spinner("Extracting Specific Information From The Text ..."):
#                     gemini_response = gemini_model.generate_content([gemini_prompt, extracted_text])

#                 # Print the Gemini response for debugging
#                 st.subheader("Gemini Response:")
#                 st.write(gemini_response)

#                 # Resolve the response to obtain the extracted specific information
#                 gemini_response.resolve()

#                 # Access the specific information from parts
#                 extracted_specific_info = get_extracted_info_from_response(gemini_response)

#                 # Display the extracted specific information
#                 st.subheader("Extracted Specific Information:")
#                 st.write(extracted_specific_info)

#                 # Append specific information to the original sheet
#         append_specific_information_to_sheet(file, image_sheet_mapping)
#     except Exception as e:
#         st.error(f"Error: {e}")

# def append_specific_information_to_sheet(excel_path, image_sheet_mapping):
#     try:
#         # Load the Excel file
#         xls = pd.ExcelFile(excel_path)

#         with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
#             for image_file, sheet_name in image_sheet_mapping.items():
#                 try:
#                     # Read the existing data from the sheet
#                     existing_data = pd.read_excel(xls, sheet_name)

#                     # Access the specific information from parts (modify as needed)
#                     specific_info = "Your_specific_information_here"

#                     # Create a DataFrame with specific information
#                     specific_info_df = pd.DataFrame({"Extracted_Specific_Info": [specific_info]})

#                     # Concatenate the existing data with specific information
#                     updated_data = pd.concat([existing_data, specific_info_df], axis=1)

#                     # Save the updated data back to the Excel file
#                     updated_data.to_excel(writer, index=False, sheet_name=sheet_name)

#                 except Exception as e:
#                     st.error(f"Error appending specific information to sheet {sheet_name}: {e}")

#     except Exception as e:
#         st.error(f"Error appending specific information to sheets: {e}")

# def get_extracted_info_from_response(response):
#     try:
#         # Access the specific information from the response
#         # Update this part based on the actual structure of the Gemini response
#         # For example, if the information is in the 'text' field, use:
#         # extracted_info = response.candidates[0].content.text
#         extracted_info = response.candidates[0].content.parts[0].text  # Update this line

#         return extracted_info
#     except Exception as e:
#         st.error(f"Error extracting specific information from Gemini response: {e}")
#         return None

# def main():
#     st.title("Excel Image and Information Extractor")

#     # File uploader
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

#     if file is not None:
#         # Extract images and process specific information from the uploaded Excel file
#         extract_images_and_process(file)

# if __name__ == "__main__":
#     main()




# import streamlit as st
# import zipfile
# from PIL import Image
# import io
# import pandas as pd
# import base64
# import google.generativeai as genai

# # Set your API keys for both Google Gemini Vision Pro and Google Gemini
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')
# gemini_model = genai.GenerativeModel('gemini-pro')

# def extract_image_from_zip(zip_ref, image_file):
#     image_data = zip_ref.read(image_file)
#     return Image.open(io.BytesIO(image_data))

# def extract_text_from_image(img):
#     vision_pro_prompt = "Extract text from the provided image."
#     with st.spinner("Extracting Text From The Image..."):
#         vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])

#     try:
#         vision_pro_response.resolve()
#         extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
#         return extracted_text
#     except Exception as e:
#         st.error(f"Error extracting text from image: {e}")
#         return None

# def extract_specific_information(extracted_text):
#     gemini_prompt = """
#                 Extract specific information from the provided text regarding Basic Terms, including:
#                 - Date Announced
#                 - Actual Completion Date
#                 - Type of Consideration
#                 - Consideration Terms
#                 - Financing Condition
#                 - Jurisdiction
#                 - Initial Expected Completion Timeline
#                 - Industry
#                 - Marketing Period
#                 - Go-Shop
#                 ignore other information & dont put any special characters like * etc
#                 """

#     with st.spinner("Extracting Specific Information From The Text ..."):
#         gemini_response = gemini_model.generate_content([gemini_prompt, extracted_text])

#     try:
#         gemini_response.resolve()
#         extracted_specific_info = get_extracted_info_from_response(gemini_response)
#         return extracted_specific_info
#     except Exception as e:
#         st.error(f"Error extracting specific information from Gemini response: {e}")
#         return None

# def extract_images_and_process(file):
#     try:
#         image_sheet_mapping = {}

#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             for image_file in image_files:
#                 sheet_name = image_file.split('/')[-1].split('.')[0]
#                 image_sheet_mapping[image_file] = sheet_name

#                 try:
#                     img = extract_image_from_zip(zip_ref, image_file)
#                     extracted_text = extract_text_from_image(img)
#                     extracted_specific_info = extract_specific_information(extracted_text)
#                     append_specific_information_to_sheet(file, sheet_name, extracted_specific_info)

#                 except Exception as e:
#                     st.error(f"Error processing sheet {sheet_name}: {e}")

#         append_specific_information_to_sheet(file, image_sheet_mapping)

#     except Exception as e:
#         st.error(f"Error: {e}")

# def append_specific_information_to_sheet(excel_path, sheet_name, specific_info):
#     try:
#         xls = pd.ExcelFile(excel_path)
#         existing_data = pd.read_excel(xls, sheet_name)
#         specific_info_df = pd.DataFrame({"Extracted_Specific_Info": [specific_info]})
#         updated_data = pd.concat([existing_data, specific_info_df], axis=1)

#         with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
#             updated_data.to_excel(writer, index=False, sheet_name=sheet_name)

#     except Exception as e:
#         st.error(f"Error appending specific information to sheet {sheet_name}: {e}")

# def get_extracted_info_from_response(response):
#     try:
#         extracted_info = response.candidates[0].content.parts[0].text
#         return extracted_info
#     except Exception as e:
#         st.error(f"Error extracting specific information from Gemini response: {e}")
#         return None

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

#     if file is not None:
#         extract_images_and_process(file)
#         st.success("Processing completed. You can download the updated Excel file below.")
#         st.download_button(
#             label="Download Excel File",
#             data=get_binary_data(file),
#             key="download_button"
#         )

# def get_binary_data(file):
#     with open(file.name, 'rb') as f:
#         data = f.read()
#     return data

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import zipfile
# from PIL import Image
# import io
# import pandas as pd
# import google.generativeai as genai

# # Set your API keys for both Google Gemini Vision Pro and Google Gemini
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')
# gemini_model = genai.GenerativeModel('gemini-pro')

# # Declare image_sheet_mapping as a global variable
# image_sheet_mapping = {}

# def extract_image_from_zip(zip_ref, image_file):
#     image_data = zip_ref.read(image_file)
#     return Image.open(io.BytesIO(image_data))

# def extract_text_from_image(img):
#     vision_pro_prompt = "Extract text from the provided image."
#     with st.spinner("Extracting Text From The Image..."):
#         vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])

#     try:
#         vision_pro_response.resolve()
#         extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
#         return extracted_text
#     except Exception as e:
#         st.error(f"Error extracting text from image: {e}")
#         return None

# def extract_specific_information(extracted_text):
#     try:
#         gemini_prompt = """
#             Extract specific information from the provided text regarding Basic Terms, including:
#             - Date Announced
#             - Actual Completion Date
#             - Type of Consideration
#             - Consideration Terms
#             - Financing Condition
#             - Jurisdiction
#             - Initial Expected Completion Timeline
#             - Industry
#             - Marketing Period
#             - Go-Shop
#             ignore other information & dont put any special characters like * etc
#             """
#         with st.spinner("Extracting Specific Information From The Text ..."):
#             gemini_response = gemini_model.generate_content([gemini_prompt, extracted_text])

#         try:
#             gemini_response.resolve()

#             # Extract specific information based on the Gemini response structure
#             # Modify this part according to the actual structure of the response
#             extracted_specific_info = get_extracted_info_from_response(gemini_response)

#             return extracted_specific_info

#         except Exception as e:
#             st.error(f"Error resolving Gemini response: {e}")
#             return None

#     except Exception as e:
#         st.error(f"Error extracting specific information from Gemini: {e}")
#         return None

# def extract_images_and_process(file):
#     try:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             sheet_names = get_sheet_names_from_excel(file)

#             for i, image_file in enumerate(image_files):
#                 if i < len(sheet_names):  # Check if there are enough sheets for the images
#                     sheet_name = sheet_names[i]

#                     try:
#                         img = extract_image_from_zip(zip_ref, image_file)
#                         extracted_text = extract_text_from_image(img)
#                         extracted_specific_info = extract_specific_information(extracted_text)

#                         st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)

#                         st.write("Extracted Text:")
#                         st.write(extracted_text)

#                         st.write("Extracted Specific Information:")
#                         st.write(extracted_specific_info)

#                     except Exception as e:
#                         st.error(f"Error processing sheet {sheet_name}: {e}")
#                 else:
#                     st.warning(f"Number of sheets is less than the number of images.")

#     except Exception as e:
#         st.error(f"Error: {e}")

# def get_sheet_names_from_excel(file):
#     try:
#         xls = pd.ExcelFile(file)
#         sheet_names = xls.sheet_names
#         return sheet_names
#     except Exception as e:
#         st.error(f"Error getting sheet names: {e}")
#         return None

# def get_extracted_info_from_response(response):
#     try:
#         extracted_info = response.candidates[0].content.parts[0].text
#         return extracted_info
#     except Exception as e:
#         st.error(f"Error extracting specific information from Gemini response: {e}")
#         return None

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         extract_images_and_process(file)
#         st.success("Processing completed.")

# if __name__ == "__main__":
#     main()



import streamlit as st
import zipfile
from PIL import Image
import io
import pandas as pd
import google.generativeai as genai

# Set your API keys for both Google Gemini Vision Pro and Google Gemini
gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

genai.configure(api_key=gemini_api_key)
vision_pro_model = genai.GenerativeModel('gemini-pro-vision')
gemini_model = genai.GenerativeModel('gemini-pro')

# Declare image_sheet_mapping as a global variable
image_sheet_mapping = {}

def extract_image_from_zip(zip_ref, image_file):
    image_data = zip_ref.read(image_file)
    return Image.open(io.BytesIO(image_data))

def extract_text_from_image(img):
    vision_pro_prompt = "Extract text from the provided image."
    with st.spinner("Extracting Text From The Image..."):
        vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])

    try:
        vision_pro_response.resolve()
        extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return None

def extract_specific_information(extracted_text):
    try:
        gemini_prompt = """
            Extract specific information from the provided text regarding Basic Terms, including:
            - Date Announced
            - Actual Completion Date
            - Type of Consideration
            - Consideration Terms
            - Financing Condition
            - Jurisdiction
            - Initial Expected Completion Timeline
            - Industry
            - Marketing Period
            - Go-Shop
            ignore other information & dont put any special characters like * etc
            """
        with st.spinner("Extracting Specific Information From The Text ..."):
            gemini_response = gemini_model.generate_content([gemini_prompt, extracted_text])

        try:
            gemini_response.resolve()

            # Extract specific information based on the Gemini response structure
            # Modify this part according to the actual structure of the response
            extracted_specific_info = get_extracted_info_from_response(gemini_response)

            return extracted_specific_info

        except Exception as e:
            st.error(f"Error resolving Gemini response: {e}")
            return None

    except Exception as e:
        st.error(f"Error extracting specific information from Gemini: {e}")
        return None

def extract_images_and_process(file):
    extracted_info_list = []

    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            media_folder = 'xl/media/'
            image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

            sheet_names = get_sheet_names_from_excel(file)

            for i, image_file in enumerate(image_files):
                if i < len(sheet_names):  # Check if there are enough sheets for the images
                    sheet_name = sheet_names[i]

                    try:
                        img = extract_image_from_zip(zip_ref, image_file)
                        extracted_text = extract_text_from_image(img)
                        extracted_specific_info = extract_specific_information(extracted_text)

                        # Append the information to the list as a dictionary
                        extracted_info_list.append({
                            "Sheet": sheet_name,
                            "Image": image_file,
                            "Extracted_Text": extracted_text,
                            "Extracted_Specific_Info": extracted_specific_info
                        })

                        st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)

                        st.write("Extracted Text:")
                        st.write(extracted_text)

                        st.write("Extracted Specific Information:")
                        st.write(extracted_specific_info)

                    except Exception as e:
                        st.error(f"Error processing sheet {sheet_name}: {e}")
                else:
                    st.warning(f"Number of sheets is less than the number of images.")

        extracted_info_df = pd.DataFrame(extracted_info_list)  # Convert the list to a DataFrame
        return extracted_info_df  # Return the DataFrame

    except Exception as e:
        st.error(f"Error: {e}")
        return None

def get_sheet_names_from_excel(file):
    try:
        xls = pd.ExcelFile(file)
        sheet_names = xls.sheet_names
        return sheet_names
    except Exception as e:
        st.error(f"Error getting sheet names: {e}")
        return None

def get_extracted_info_from_response(response):
    try:
        extracted_info = response.candidates[0].content.parts[0].text
        return extracted_info
    except Exception as e:
        st.error(f"Error extracting specific information from Gemini response: {e}")
        return None

def get_binary_data(data, file_name="output_extracted_info.xlsx"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name='Extracted_Information', index=False)
    output.seek(0)
    return output.getvalue(), file_name

def main():
    st.title("Excel Image and Information Extractor")
    file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

    if file is not None:
        extracted_info_df = extract_images_and_process(file)

        if extracted_info_df is not None:
            st.success("Processing completed. You can download the updated Excel file with extracted information below.")
            binary_data, file_name = get_binary_data(extracted_info_df)
            st.download_button(
                label="Download Excel File",
                data=binary_data,
                key="download_button",
                file_name=file_name,
            )

if __name__ == "__main__":
    main()
