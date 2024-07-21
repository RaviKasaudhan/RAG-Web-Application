# RAG Web Application

## Introduction

The RAG Web Application leverages Azure and OpenAI services to process and index documents, providing summarized answers using the "text-embedding-ada-003" model. This also give user the option to fetch the important key value pairs from the documents and the whole summary of the document. 

## Getting Started

### Prerequisites

To use this application, you will need the following:

1. **Azure Blob Storage**:
   - **Two Blob Containers**: 
     - One for storing your files.
     - One for storing the index of your files.

2. **Credentials**:
   - **OpenAI**: For utilizing the "text-embedding-ada-003" model.
   - **Azure Form Recognizer**: For extracting data from documents.

### Setup Instructions

1. **Clone the Repository**:
   First, clone this repository to your local directory.
   ```bash
   git clone https://github.com/YourUsername/RAG-Web-Application.git
   
2. **Azure Blob Containers:**

   a. Create two Azure Blob containers in your Azure account.
      Files Container: To store your files.
      Index Container: To store the index of your files.
   
3. **Configure Credentials:**
   Ensure you have the following credentials configured:
      OpenAI API Key
      Azure Form Recognizer API Key
      Azure Blob Storage Connection String
   
3. **Set Up Azure Function App:**

   a. Navigate to the function_app folder in the repository.
   b. Create an HTTP trigger function app in Azure.
   c. Upload the contents of the function_app folder to your Azure Function App.
   d. This function will be triggered once a user uploads a file. It will create indexes and store them in the index container in Azure.

**Usage**
1. **Upload Files:**

   Upload your files to the designated Azure Blob container for files.
   
2. **Indexing:**

   The function app will automatically index the files and store the indices in the designated Azure Blob container for indexing.

3. **Summarization:**

   Use the application to query and retrieve summarized answers based on the indexed documents.

**Install Dependencies:**
   Ensure you have Python and pip installed. Then, install the requiremnts.txt file with command 
   ```bash
   pip install -r requirements.txt
   ```

**Run the application with:**

   ```bash
   streamlit run app.py

