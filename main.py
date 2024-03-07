import streamlit as st
import pandas as pd
from PIL import Image
import base64
import google.generativeai as genai


# Set your API keys for both Google Gemini Vision Pro and Google Gemini
gemini_api_key = "AIzaSyAFER-GEGVy5Cw9E-vkCIjyjvW-Bc4pBZ8"

genai.configure(api_key=gemini_api_key)
# Create an instance of the GenerativeModel class with the model name 'gemini-pro-vision' and set the API key
vision_pro_model = genai.GenerativeModel('gemini-pro-vision')

# Create an instance of the GenerativeModel class with the model name 'gemini-pro' and set the API key
gemini_model = genai.GenerativeModel('gemini-pro')

# Streamlit app
def main():
    st.title("Image To Excel")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open the uploaded image using the Pillow library
        image_pil = Image.open(uploaded_image)

        # Display the uploaded image
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)

        # Prompt for extracting text using Gemini Vision Pro
        vision_pro_prompt = "Extract text from the provided image."

        # Generate content using the Google Gemini Vision Pro API to extract text from the image
        with st.spinner("Extracting Text From The Image..."):
            vision_pro_response = vision_pro_model.generate_content([vision_pro_prompt, image_pil])

        # Resolve the response to obtain the extracted text
        vision_pro_response.resolve()

        # Access the text from parts
        extracted_text = " ".join(part.text for part in vision_pro_response.candidates[0].content.parts)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.write(extracted_text)

        # Prompt for extracting specific information using Gemini Pro
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

        # Use the Google Gemini API to extract specific information from the text
        with st.spinner("Extracting Specific Information From The Text ..."):
            gemini_response = extract_specific_information(extracted_text, gemini_api_key, gemini_prompt)

        # Display the result from Google Gemini API
        st.subheader("Extracted Specific Information:")
        st.write(gemini_response.text)

        # Create DataFrame and Export to Excel
        df = create_dataframe(gemini_response.text)

        # Specify the file path for the Excel file
        excel_file_path = "output.xlsx"

        # Increase the column widths
        col_widths = [max(len(str(value)) + 4, 20) for value in df.iloc[0]]

        # Create the Excel file with increased column widths
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']
            for i, width in enumerate(col_widths):
                worksheet.column_dimensions[worksheet.cell(row=1, column=i + 1).column_letter].width = width

        # Encode the Excel data to base64 for download
        with open(excel_file_path, 'rb') as file:
            excel_data = file.read()
            b64 = base64.b64encode(excel_data).decode()

        # Create a download link
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{excel_file_path}">Click here to download the Excel file</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Display the styled DataFrame
        st.write(df)

def extract_specific_information(text, api_key, prompt):
    # Use the Google Gemini API to extract specific information
    gemini_response = gemini_model.generate_content([prompt, text])
    gemini_response.resolve()

    return gemini_response

def create_dataframe(text):
    # Parse the extracted text to create a DataFrame
    lines = text.split("\n")
    data = {}

    for line in lines:
        if ":" in line:
            key, value = map(str.strip, line.split(":", 1))
            # Replace invalid characters in column names
            key = key.replace("=", "").replace("-", "").strip()
            data[key] = [value]

    # Create a DataFrame with proper columns and values
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    main()

