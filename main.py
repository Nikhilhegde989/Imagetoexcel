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
#     extracted_info_list = []

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

#                         # Append the information to the list as a dictionary
#                         extracted_info_list.append({
#                             "Sheet": sheet_name,
#                             "Image": image_file,
#                             "Extracted_Text": extracted_text,
#                             "Extracted_Specific_Info": extracted_specific_info
#                         })

#                         st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)

#                         st.write("Extracted Text:")
#                         st.write(extracted_text)

#                         st.write("Extracted Specific Information:")
#                         st.write(extracted_specific_info)

#                     except Exception as e:
#                         st.error(f"Error processing sheet {sheet_name}: {e}")
#                 else:
#                     st.warning(f"Number of sheets is less than the number of images.")

#         extracted_info_df = pd.DataFrame(extracted_info_list)  # Convert the list to a DataFrame
#         return extracted_info_df  # Return the DataFrame

#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None

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

# def get_binary_data(data, file_name="output_extracted_info.xlsx", original_file=None):
#     try:
#         if original_file is not None:
#             with pd.ExcelWriter(original_file, engine='openpyxl', mode='a') as writer:
#                 data.to_excel(writer, sheet_name='Extracted_Information', index=False)
#             return None, None
#         else:
#             output = io.BytesIO()
#             with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#                 data.to_excel(writer, sheet_name='Extracted_Information', index=False)
#             output.seek(0)
#             return output.getvalue(), file_name
#     except FileNotFoundError:
#         st.error("File not found. Please check the file path or name.")
#         return None, None
#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None, None


# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         extracted_info_df = extract_images_and_process(file)

#         if extracted_info_df is not None:
#             st.success("Processing completed. You can download the updated Excel file with extracted information below.")

#             # Corrected the use of file.name
#             original_file_name = file.name if isinstance(file, io.BytesIO) else None

#             binary_data, file_name = get_binary_data(extracted_info_df, original_file=original_file_name)
#             st.download_button(
#                 label="Download Excel File",
#                 data=binary_data,
#                 key="download_button",
#                 file_name=file_name,
#             )

# if __name__ == "__main__":
#     main()





# import streamlit as st
# import zipfile
# from PIL import Image
# import openpyxl
# import os
# import io
# import base64
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

# def extract_images_and_process(file, current_sheet):
#     try:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Create the "images" subfolder if it doesn't exist
#             os.makedirs("images", exist_ok=True)

#             with pd.ExcelWriter("output_extracted_info.xlsx", engine='openpyxl') as writer:
#                 for i, image_file in enumerate(image_files):
#                     if i < len(current_sheet):  # Check if there are enough sheets for the images
#                         sheet_name = current_sheet[i]

#                         try:
#                             img = extract_image_from_zip(zip_ref, image_file)
#                             extracted_text = extract_text_from_image(img)
#                             extracted_specific_info = extract_specific_information(extracted_text)

#                             # Save the image to the "images" subfolder
#                             img_path = f"images/{image_file.split('/')[-1]}"  # Extract the filename
#                             img.save(img_path)

#                             # Write the DataFrame to a separate sheet in the Excel file
#                             extracted_info_df = pd.DataFrame({
#                                 "Sheet": [sheet_name],
#                                 "Image": [img_path],  # Save the image path instead of file name
#                                 "Extracted_Text": [extracted_text],
#                                 "Extracted_Specific_Info": [extracted_specific_info]
#                             })

#                             extracted_info_df.to_excel(writer, sheet_name=sheet_name, index=False)

#                             # Insert the image into the Excel sheet at a specific location
#                             sheet = writer.sheets[sheet_name]
#                             img_ref = openpyxl.drawing.image.Image(img_path)
#                             sheet.add_image(img_ref, 'A1')  # Adjust the location as needed

#                             st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)
#                             st.write("Extracted Text:")
#                             st.write(extracted_text)
#                             st.write("Extracted Specific Information:")
#                             st.write(extracted_specific_info)

#                         except Exception as e:
#                             st.error(f"Error processing sheet {sheet_name}: {e}")
#                     else:
#                         st.warning(f"Number of sheets is less than the number of images.")

#             st.success("Processing completed. You can download the updated Excel file with extracted information below.")

#             # Provide a download link for the updated Excel file
#             st.markdown(get_download_link("output_extracted_info.xlsx"), unsafe_allow_html=True)
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

# def get_download_link(file_path):
#     with open(file_path, "rb") as f:
#         file_content = f.read()
#     return f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_content).decode()}" download="{file_path}">Download Excel File</a>'

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         current_sheet = get_sheet_names_from_excel(file)
#         extract_images_and_process(file, current_sheet)
# if __name__ == "__main__":
#     main()







# import streamlit as st
# import zipfile
# from PIL import Image
# import openpyxl
# import os
# import io
# import base64
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
#     vision_pro_prompt = """
#     Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image.It contains 3 columns only 'Antitrust','SEC','All Other Terms' in the tabular format same as in the image. 
#     below  is an example(its just an example give with actual data)
#     "| Regulatory Timeline & Events|
#     |----|
#     |Antitrust|SEC|All Other Terms|
#     |DOJ Approval(consent) 10/10/2018|AET SH Approval 3/13/2018|Walk Date 12/03/2018|
#     |HSR Second Request Issued 02/01/2018|CVS SH Approval 03/13/2018|Expected Close 11/28/2018|
#     ||Definitive field 02/09/2018|NY DOI Approval 11/26/2018|
#     |||NJ DOI Approval 11/21/2018|
#     etc...."
#     The actual number of rows  may vary some column can be empty also.Overall idea is to  Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image in a tabular format without missing any data
#     """ 
#     with st.spinner("Extracting Text From The Image..."):
#         vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])
#         print(vision_pro_response)

#     try:
#         vision_pro_response.resolve()
#         extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
#         print(extracted_text)
#         return extracted_text
#     except Exception as e:
#         st.error(f"Error extracting text from image: {e}")
#         return None

# def extract_specific_information(extracted_text):
#     try:
#         gemini_prompt = """
#             The text contains information which should be divided into 3 columns:
#             1)Antitrust
#             2)SEC
#             3)All other terms
#             all the text are in serial order you have to rearrange that.
#             Example "Regulatory Timeline & Events Antitrust SEC All Other Items 08/04/2022 Walk Date 06/16/2022 Definitive Filed 10/18/2022 Expected Close 05/10/2022 Preliminary Proxy Filed 08/09/2022" 
#             this text will be 
#             Antitrust           SEC            All Other Items
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

# def extract_images_and_process(file, current_sheet):
#     try:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Create the "images" subfolder if it doesn't exist
#             os.makedirs("images", exist_ok=True)

#             with pd.ExcelWriter("output_extracted_info.xlsx", engine='openpyxl') as writer:
#                 for i, image_file in enumerate(image_files):
#                     if i < len(current_sheet):  # Check if there are enough sheets for the images
#                         sheet_name = current_sheet[i]

#                         try:
#                             img = extract_image_from_zip(zip_ref, image_file)
#                             extracted_text = extract_text_from_image(img)
#                             extracted_specific_info = extract_specific_information(extracted_text)

#                             # Save the image to the "images" subfolder
#                             img_path = f"images/{image_file.split('/')[-1]}"  # Extract the filename
#                             img.save(img_path)

#                             # Write the DataFrame to a separate sheet in the Excel file
#                             extracted_info_df = pd.DataFrame({
#                                 "Sheet": [sheet_name],
#                                 "Image": [img_path],  # Save the image path instead of file name
#                                 "Extracted_Text": [extracted_text],
#                                 "Extracted_Specific_Info": [extracted_specific_info]
#                             })

#                             extracted_info_df.to_excel(writer, sheet_name=sheet_name, index=False)

#                             # Insert the image into the Excel sheet at a specific location
#                             sheet = writer.sheets[sheet_name]
#                             img_ref = openpyxl.drawing.image.Image(img_path)
#                             sheet.add_image(img_ref, 'A1')  # Adjust the location as needed

#                             st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)
#                             st.write("Extracted Text:")
#                             st.write(extracted_text)
#                             st.write("Extracted Specific Information:")
#                             st.write(extracted_specific_info)

#                         except Exception as e:
#                             st.error(f"Error processing sheet {sheet_name}: {e}")
#                     else:
#                         st.warning(f"Number of sheets is less than the number of images.")

#             st.success("Processing completed. You can download the updated Excel file with extracted information below.")

#             # Provide a download link for the updated Excel file
#             st.markdown(get_download_link("output_extracted_info.xlsx"), unsafe_allow_html=True)
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

# def get_download_link(file_path):
#     with open(file_path, "rb") as f:
#         file_content = f.read()
#     return f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_content).decode()}" download="{file_path}">Download Excel File</a>'

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         current_sheet = get_sheet_names_from_excel(file)
#         extract_images_and_process(file, current_sheet)
# if __name__ == "__main__":
#     main()




# import streamlit as st
# import zipfile
# from PIL import Image
# import openpyxl
# import os
# import io
# import base64
# import pandas as pd
# import google.generativeai as genai

# # Set your API keys for Google Gemini Vision Pro
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# # Declare image_sheet_mapping as a global variable
# image_sheet_mapping = {}

# def extract_image_from_zip(zip_ref, image_file):
#     image_data = zip_ref.read(image_file)
#     return Image.open(io.BytesIO(image_data))

# def extract_text_from_image(img):
#     vision_pro_prompt = """
#     Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image.It contains 3 columns only 'Antitrust','SEC','All Other Terms' in the tabular format same as in the image. 
#     below  is an example(its just an example give with actual data). every image will have 3 columns 'Antitrust','SEC','All Other Terms' each row contains name with date.
#     "| Regulatory Timeline & Events|
#     |----|
#     |Antitrust|SEC|All Other Terms|
#     |DOJ Approval(consent) 10/10/2018|AET SH Approval 3/13/2018|Walk Date 12/03/2018|
#     |HSR Second Request Issued 02/01/2018|CVS SH Approval 03/13/2018|Expected Close 11/28/2018|
#     ||Definitive field 02/09/2018|NY DOI Approval 11/26/2018|
#     |||NJ DOI Approval 11/21/2018|
#     etc...."
#     The actual number of rows  may vary some column can be empty also.Overall idea is to  Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image in a tabular format without missing any data
#     """ 
#     with st.spinner("Extracting Text From The Image..."):
#         vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])
#         print(vision_pro_response)

#     try:
#         vision_pro_response.resolve()
#         extracted_text = vision_pro_response.candidates[0].content.text
#         print(extracted_text)
#         return extracted_text
#     except Exception as e:
#         st.error(f"Error extracting text from image: {e}")
#         return None

# def extract_images_and_process(file, current_sheet):
#     try:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Create the "images" subfolder if it doesn't exist
#             os.makedirs("images", exist_ok=True)

#             with pd.ExcelWriter("output_extracted_info.xlsx", engine='openpyxl') as writer:
#                 for i, image_file in enumerate(image_files):
#                     if i < len(current_sheet):  # Check if there are enough sheets for the images
#                         sheet_name = current_sheet[i]

#                         try:
#                             img = extract_image_from_zip(zip_ref, image_file)
#                             extracted_text = extract_text_from_image(img)
#                             table_data = parse_response_to_table(extracted_text)

#                             # Extracted text directly inserted into its respective sheet
#                             extracted_info_df = pd.DataFrame(table_data)

#                             extracted_info_df.to_excel(writer, sheet_name=sheet_name, index=False)

#                             # Insert the image into the Excel sheet at a specific location
#                             sheet = writer.sheets[sheet_name]
#                             img_ref = openpyxl.drawing.image.Image(f"images/{image_file.split('/')[-1]}")
#                             sheet.add_image(img_ref, 'A1')  # Adjust the location as needed

#                             st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)
#                             st.write("Extracted Text:")
#                             st.write(extracted_text)

#                         except Exception as e:
#                             st.error(f"Error processing sheet {sheet_name}: {e}")
#                     else:
#                         st.warning(f"Number of sheets is less than the number of images.")

#             st.success("Processing completed. You can download the updated Excel file with extracted information below.")

#             # Provide a download link for the updated Excel file
#             st.markdown(get_download_link("output_extracted_info.xlsx"), unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Error: {e}")

# def parse_response_to_table(extracted_text):
#     rows = extracted_text.split('\n')
#     headers = rows[1].split('|')[1:-1]  # Extract column headers
#     data_rows = rows[3:-2]  # Extract data rows
#     table_data = []
#     for row in data_rows:
#         columns = row.split('|')[1:-1]
#         table_data.append(columns)
#     return {'Antitrust': [data[0] for data in table_data],
#             'SEC': [data[1] for data in table_data],
#             'All Other Terms': [data[2] for data in table_data]}

# def get_sheet_names_from_excel(file):
#     try:
#         xls = pd.ExcelFile(file)
#         sheet_names = xls.sheet_names
#         return sheet_names
#     except Exception as e:
#         st.error(f"Error getting sheet names: {e}")
#         return None

# def get_download_link(file_path):
#     with open(file_path, "rb") as f:
#         file_content = f.read()
#     return f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_content).decode()}" download="{file_path}">Download Excel File</a>'

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         current_sheet = get_sheet_names_from_excel(file)
#         extract_images_and_process(file, current_sheet)
        
# if __name__ == "__main__":
#     main()





# import streamlit as st
# import zipfile
# from PIL import Image
# import openpyxl
# import os
# import io
# import base64
# import pandas as pd
# import google.generativeai as genai

# # Set your API keys for Google Gemini Vision Pro
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# # Declare image_sheet_mapping as a global variable
# image_sheet_mapping = {}

# def extract_image_from_zip(zip_ref, image_file):
#     image_data = zip_ref.read(image_file)
#     return Image.open(io.BytesIO(image_data))

# def extract_text_from_image(img):
#     vision_pro_prompt = """
#     Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image.It contains 3 columns only 'Antitrust','SEC','All Other Terms' in the tabular format same as in the image. 
#     below  is an example(its just an example give with actual data). every image will have 3 columns 'Antitrust','SEC','All Other Terms' each row contains name with date.
#     if there is no value in that row & column just leave blank | | | |.
#     in each cell event name & date should be there
#     "
#     |Antitrust|SEC|All Other Terms|
#     |event_name date (for first row & first column)|event_name date (for first row & second column)|event_name date (for first row & third column)|
#     |event_name date (for second row & first column)|event_name date (for second row & second column)|event_name date (for second row & third column)|
#     ||event_name date (for nth row & second column)|event_name date (for nth row & third column)|
#     |||event_name date (for nth row & third column)|
#     etc...."
#     The actual number of rows  may vary some column can be empty also.Overall idea is to  Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image in a tabular format without missing any data
#     """ 
#     with st.spinner("Extracting Text From The Image..."):
#         vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])
#         print(vision_pro_response)

#     try:
#         vision_pro_response.resolve()
#         extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
#         print(extracted_text)
#         return extracted_text
#     except Exception as e:
#         st.error(f"Error extracting text from image: {e}")
#         return None

# def extract_images_and_process(file, current_sheet):
#     try:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Create the "images" subfolder if it doesn't exist
#             os.makedirs("images", exist_ok=True)

#             with pd.ExcelWriter("output_extracted_info.xlsx", engine='openpyxl') as writer:
#                 for i, image_file in enumerate(image_files):
#                     if i < len(current_sheet):  # Check if there are enough sheets for the images
#                         sheet_name = current_sheet[i]

#                         try:
#                             img = extract_image_from_zip(zip_ref, image_file)
#                             extracted_text = extract_text_from_image(img)

#                             # Extracted text directly inserted into its respective sheet
#                             extracted_info_df = pd.DataFrame({
#                                 "Sheet": [sheet_name],
#                                 "Image": [f"images/{image_file.split('/')[-1]}"],  # Save the image path instead of file name
#                                 "Extracted_Text": [extracted_text]
#                             })

#                             extracted_info_df.to_excel(writer, sheet_name=sheet_name, index=False)

#                             # Insert the image into the Excel sheet at a specific location
#                             sheet = writer.sheets[sheet_name]
#                             img_ref = openpyxl.drawing.image.Image(f"images/{image_file.split('/')[-1]}")
#                             sheet.add_image(img_ref, 'A1')  # Adjust the location as needed

#                             st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)
#                             st.write("Extracted Text:")
#                             st.write(extracted_text)

#                         except Exception as e:
#                             st.error(f"Error processing sheet {sheet_name}: {e}")
#                     else:
#                         st.warning(f"Number of sheets is less than the number of images.")

#             st.success("Processing completed. You can download the updated Excel file with extracted information below.")

#             # Provide a download link for the updated Excel file
#             st.markdown(get_download_link("output_extracted_info.xlsx"), unsafe_allow_html=True)
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

# def get_download_link(file_path):
#     with open(file_path, "rb") as f:
#         file_content = f.read()
#     return f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_content).decode()}" download="{file_path}">Download Excel File</a>'

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         current_sheet = get_sheet_names_from_excel(file)
#         extract_images_and_process(file, current_sheet)
        
# if __name__ == "__main__":
#     main()



# import streamlit as st
# import zipfile
# from PIL import Image
# import openpyxl
# import os
# import io
# import base64
# import pandas as pd
# import google.generativeai as genai

# # Set your API keys for Google Gemini Vision Pro
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# # Declare image_sheet_mapping as a global variable
# image_sheet_mapping = {}

# def extract_image_from_zip(zip_ref, image_file):
#     image_data = zip_ref.read(image_file)
#     return Image.open(io.BytesIO(image_data))

# def extract_text_from_image(img):
#     vision_pro_prompt = """
#     Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image.It contains 3 columns only 'Antitrust','SEC','All Other Terms' in the tabular format same as in the image. 
#     below  is an example(its just an example give with actual data). every image will have 3 columns 'Antitrust','SEC','All Other Terms' each row contains name with date.
#     if there is no value in that row & column just leave blank | | | |.
#     in each cell event name & date should be there
#     "
#     |Antitrust|SEC|All Other Terms|
#     |event_name date (for first row & first column)|event_name date (for first row & second column)|event_name date (for first row & third column)|
#     |event_name date (for second row & first column)|event_name date (for second row & second column)|event_name date (for second row & third column)|
#     ||event_name date (for nth row & second column)|event_name date (for nth row & third column)|
#     |||event_name date (for nth row & third column)|
#     etc...."
#     The actual number of rows  may vary some column can be empty also.Overall idea is to  Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image in a tabular format without missing any data
#     """ 
#     with st.spinner("Extracting Text From The Image..."):
#         vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])
#         print(vision_pro_response)

#     try:
#         vision_pro_response.resolve()
#         extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
#         print(extracted_text)
#         return extracted_text
#     except Exception as e:
#         st.error(f"Error extracting text from image: {e}")
#         return None

# def extract_images_and_process(file, current_sheet):
#     try:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Create the "images" subfolder if it doesn't exist
#             os.makedirs("images", exist_ok=True)

#             with pd.ExcelWriter("output_extracted_info.xlsx", engine='openpyxl') as writer:
#                 for i, image_file in enumerate(image_files):
#                     if i < len(current_sheet):  # Check if there are enough sheets for the images
#                         sheet_name = current_sheet[i]

#                         try:
#                             img = extract_image_from_zip(zip_ref, image_file)
#                             extracted_text = extract_text_from_image(img)

#                             # Split extracted text into rows and columns
#                             rows = extracted_text.split('\n')
#                             data = []
#                             for row in rows:
#                                 data.append(row.split('|')[1:-1])  # Remove first and last empty items

#                             # Write the DataFrame to a separate sheet in the Excel file
#                             extracted_info_df = pd.DataFrame(data, columns=["Antitrust", "SEC", "All Other Terms"])
#                             extracted_info_df.to_excel(writer, sheet_name=sheet_name, index=False)

#                             # Insert the image into the Excel sheet at a specific location
#                             sheet = writer.sheets[sheet_name]
#                             img_ref = openpyxl.drawing.image.Image(f"images/{image_file.split('/')[-1]}")
#                             sheet.add_image(img_ref, 'A1')  # Adjust the location as needed

#                             st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)
#                             st.write("Extracted Text:")
#                             st.write(extracted_text)

#                         except Exception as e:
#                             st.error(f"Error processing sheet {sheet_name}: {e}")
#                     else:
#                         st.warning(f"Number of sheets is less than the number of images.")

#             st.success("Processing completed. You can download the updated Excel file with extracted information below.")

#             # Provide a download link for the updated Excel file
#             st.markdown(get_download_link("output_extracted_info.xlsx"), unsafe_allow_html=True)
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

# def get_download_link(file_path):
#     with open(file_path, "rb") as f:
#         file_content = f.read()
#     return f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_content).decode()}" download="{file_path}">Download Excel File</a>'

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         current_sheet = get_sheet_names_from_excel(file)
#         extract_images_and_process(file, current_sheet)
        
# if __name__ == "__main__":
#     main()




# import streamlit as st
# import zipfile
# from PIL import Image
# import openpyxl
# import os
# import io
# import base64
# import pandas as pd
# import google.generativeai as genai

# # Set your API keys for Google Gemini Vision Pro
# gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

# genai.configure(api_key=gemini_api_key)
# vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# # Declare image_sheet_mapping as a global variable
# image_sheet_mapping = {}

# def extract_image_from_zip(zip_ref, image_file):
#     image_data = zip_ref.read(image_file)
#     return Image.open(io.BytesIO(image_data))

# def extract_text_from_image(img):
#     vision_pro_prompt = """
#     Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image.It contains 3 columns only 'Antitrust','SEC','All Other Terms' in the tabular format same as in the image. 
#     below  is an example(its just an example give with actual data). every image will have 3 columns 'Antitrust','SEC','All Other Terms' each row contains name with date.
#     if there is no value in that row & column just leave blank | | | |.
#     in each cell event name & date should be there. Strictly 3 columns only.
#     |Antitrust|SEC|All Other Terms|
#     |event_name date (for first row & first column)|event_name date (for first row & second column)|event_name date (for first row & third column)|
#     |event_name date (for second row & first column)|event_name date (for second row & second column)|event_name date (for second row & third column)|
#     ||event_name date (for nth row & second column)|event_name date (for nth row & third column)|
#     |||event_name date (for nth row & third column)|
#     etc....
#     The actual number of rows  may vary some column can be empty also.Overall idea is to  Extract text which is after the heading 'Regulatory Timeline & Events'from the provided image in a tabular format without missing any data
#     """ 
#     with st.spinner("Extracting Text From The Image..."):
#         vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])
#         print(vision_pro_response)

#     try:
#         vision_pro_response.resolve()
#         extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
#         print(extracted_text)
#         return extracted_text
#     except Exception as e:
#         st.error(f"Error extracting text from image: {e}")
#         return None

# def extract_images_and_process(file, current_sheet):
#     try:
#         with zipfile.ZipFile(file, 'r') as zip_ref:
#             media_folder = 'xl/media/'
#             image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

#             # Create the "images" subfolder if it doesn't exist
#             os.makedirs("images", exist_ok=True)

#             with pd.ExcelWriter("output_extracted_info.xlsx", engine='openpyxl') as writer:
#                 for i, image_file in enumerate(image_files):
#                     if i < len(current_sheet):  # Check if there are enough sheets for the images
#                         sheet_name = current_sheet[i]

#                         try:
#                             img = extract_image_from_zip(zip_ref, image_file)
#                             extracted_text = extract_text_from_image(img)

#                             # Split extracted text into rows and columns
#                             rows = extracted_text.split('\n')
#                             data = []
#                             for row in rows:
#                                 columns = row.split('|')[1:-1]  # Remove first and last empty items
#                                 if len(columns) == 3:  # Ensure the correct number of columns
#                                     data.append(columns)
#                                 else:
#                                     st.error(f"Incorrect number of columns in extracted text: {row}")

#                             # Write the DataFrame to a separate sheet in the Excel file
#                             if data:
#                                 extracted_info_df = pd.DataFrame(data, columns=["Antitrust", "SEC", "All Other Terms"])
#                                 extracted_info_df.to_excel(writer, sheet_name=sheet_name, index=False)

#                                 # Insert the image into the Excel sheet at a specific location
#                                 sheet = writer.sheets[sheet_name]
#                                 img_ref = openpyxl.drawing.image.Image(f"images/{image_file.split('/')[-1]}")
#                                 sheet.add_image(img_ref, 'A1')  # Adjust the location as needed

#                                 st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)
#                                 st.write("Extracted Text:")
#                                 st.write(extracted_text)
#                             else:
#                                 st.warning("No valid data extracted from the image.")

#                         except Exception as e:
#                             st.error(f"Error processing sheet {sheet_name}: {e}")
#                     else:
#                         st.warning(f"Number of sheets is less than the number of images.")

#             st.success("Processing completed. You can download the updated Excel file with extracted information below.")

#             # Provide a download link for the updated Excel file
#             st.markdown(get_download_link("output_extracted_info.xlsx"), unsafe_allow_html=True)
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

# def get_download_link(file_path):
#     with open(file_path, "rb") as f:
#         file_content = f.read()
#     return f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_content).decode()}" download="{file_path}">Download Excel File</a>'

# def main():
#     st.title("Excel Image and Information Extractor")
#     file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

#     if file is not None:
#         current_sheet = get_sheet_names_from_excel(file)
#         extract_images_and_process(file, current_sheet)
        
# if __name__ == "__main__":
#     main()


import streamlit as st
import zipfile
from PIL import Image
import openpyxl
import os
import io
import base64
import pandas as pd
import google.generativeai as genai

# Set your API keys for Google Gemini Vision Pro
gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

genai.configure(api_key=gemini_api_key)
vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# Declare image_sheet_mapping as a global variable
image_sheet_mapping = {}

def extract_image_from_zip(zip_ref, image_file):
    image_data = zip_ref.read(image_file)
    return Image.open(io.BytesIO(image_data))

def extract_text_from_image(img):
    vision_pro_prompt = """
    Extract text from the image following these guidelines:
- The table should contain three columns: 'Antitrust', 'SEC', and 'All Other Terms'.
- Each cell definately contain | eventname & date |
- Each row should consist of pairs of event names and dates, separated by '|' within each column.
- If an event name or date is missing, leave the cell blank (use '| |' to represent an empty cell).
- Example format:
    |Antitrust          |SEC                  |All Other Terms             |
    |event_name1 date1  |event_name2 date2    |event_name3 date3          |
    |event_name4 date4  |                     |event_name5 date5          |
    |                    |event_name6 date6    |event_name7 date7          |
    |event_name8 date8  |event_name9 date9    |                           |
    |                    |                     |event_name10 date10        |
    |event_name11 date11|                     |                           |
    """ 
    with st.spinner("Extracting Text From The Image..."):
        vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, img])
        print(vision_pro_response)

    try:
        vision_pro_response.resolve()
        extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)
        print(extracted_text)
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return None

def extract_images_and_process(file, current_sheet):
    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            media_folder = 'xl/media/'
            image_files = [name for name in zip_ref.namelist() if name.startswith(media_folder)]

            # Create the "images" subfolder if it doesn't exist
            os.makedirs("images", exist_ok=True)

            with pd.ExcelWriter("output_extracted_info.xlsx", engine='openpyxl') as writer:
                for i, image_file in enumerate(image_files):
                    if i < len(current_sheet):  # Check if there are enough sheets for the images
                        sheet_name = current_sheet[i]

                        try:
                            img = extract_image_from_zip(zip_ref, image_file)
                            extracted_text = extract_text_from_image(img)

                            # Split extracted text into rows and columns
                            rows = extracted_text.split('\n')
                            data = []
                            for row in rows:
                                if row.strip():  # Check if the row is not empty
                                    columns = row.split('|')[1:-1]  # Remove first and last empty items
                                    if len(columns) == 3:  # Ensure the correct number of columns
                                        data.append(columns)
                                    else:
                                        st.warning(f"Skipping row with incorrect number of columns: {row}")


                            # Write the DataFrame to a separate sheet in the Excel file
                            if data:
                                extracted_info_df = pd.DataFrame(data, columns=["Antitrust", "SEC", "All Other Terms"])
                                extracted_info_df.to_excel(writer, sheet_name=sheet_name, index=False)

                                # Insert the image into the Excel sheet at a specific location
                                sheet = writer.sheets[sheet_name]
                                img_ref = openpyxl.drawing.image.Image(f"images/{image_file.split('/')[-1]}")
                                sheet.add_image(img_ref, 'F1')  # Adjust the location as needed

                                st.image(img, caption=f"Sheet Name: {sheet_name}, Image File: {image_file}", use_column_width=True)
                                st.write("Extracted Text:")
                                st.write(extracted_text)
                            else:
                                st.warning("No valid data extracted from the image.")

                        except Exception as e:
                            st.error(f"Error processing sheet {sheet_name}: {e}")
                    else:
                        st.warning(f"Number of sheets is less than the number of images.")

            st.success("Processing completed. You can download the updated Excel file with extracted information below.")

            # Provide a download link for the updated Excel file
            st.markdown(get_download_link("output_extracted_info.xlsx"), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")

def get_sheet_names_from_excel(file):
    try:
        xls = pd.ExcelFile(file)
        sheet_names = xls.sheet_names
        return sheet_names
    except Exception as e:
        st.error(f"Error getting sheet names: {e}")
        return None

def get_download_link(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()
    return f'<a href="data:application/octet-stream;base64,{base64.b64encode(file_content).decode()}" download="{file_path}">Download Excel File</a>'

def main():
    st.title("Excel Image and Information Extractor")
    file = st.file_uploader("Upload Excel file", type=["xlsx", "xls", "zip"])

    if file is not None:
        current_sheet = get_sheet_names_from_excel(file)
        extract_images_and_process(file, current_sheet)
        
if __name__ == "__main__":
    main()
