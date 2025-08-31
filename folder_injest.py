import os
import pickle
from llama_parse import LlamaParse
llamaparse_api_key = "llx-WZmMHwGdlNPF8UG2ZZtddEjS7fhTS7uv9ieQFtSjqqTEwBnA"
# Function to load or parse data
def load_or_parse_data(pdf_path, output_folder):
    # Create a folder for parsed data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract filename without extension
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Define the output folder for the PDF's parsed data
    pdf_output_folder = os.path.join(output_folder, pdf_filename)
    if not os.path.exists(pdf_output_folder):
        os.makedirs(pdf_output_folder)
    
    # Define the data file path
    data_file = os.path.join(pdf_output_folder, f"{pdf_filename}.pkl")
    
    if os.path.exists(data_file):
        # Load the parsed data from the file
        with open(data_file, "rb") as f:
            parsed_data = pickle.load(f)
    else:
        # Define parsing instruction with dynamic PDF file name
        parsing_instruction = f"""The provided document, {pdf_filename}, is crucial as it contains all the pertinent information
        Please extract all the information from the file in a structured format"""
        
        # Perform the parsing step
        parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", parsing_instruction=parsing_instruction)
        llama_parse_documents = parser.load_data(pdf_path)
        
        # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(llama_parse_documents, f)
        
        # Set the parsed data to the variable
        parsed_data = llama_parse_documents
    
    return parsed_data

# Define input and output folders
pdf_folder = "PDFs"
markdown_folder = "Markdown-Factsheet"
parsed_data_folder = "pkl_Factsheets"

# Process each PDF file
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        
        # Load or parse the data for each PDF file
        parsed_data = load_or_parse_data(pdf_path, parsed_data_folder)
        
        # Save the parsed data as Markdown file
        markdown_filename = os.path.splitext(pdf_file)[0] + ".md"
        markdown_path = os.path.join(markdown_folder, markdown_filename)
        with open(markdown_path, 'w') as f:
            for doc in parsed_data:
                f.write(doc.text)
                print(doc.text)
