import streamlit as st
from dataclasses import dataclass
import csv, html, json, logging, openai, os, nltk.data, time, traceback
from azure.storage.blob import BlobServiceClient
from langchain.document_loaders import UnstructuredPDFLoader
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import requests
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.llms import AzureOpenAI
import pandas as pd
from langchain.chains import RetrievalQA
import shutil


def config_setting(ROOT_DIR,home):
    try:
        logging.info('Code running from: {}'.format(ROOT_DIR))
        logging.info('Home Path: {}'.format(home))

        with open(os.path.join(ROOT_DIR, 'config.json'), 'r') as jsonfile:
            config = json.load(jsonfile)
        oai_base = config['openai']['base']
        oai_key = config['openai']['key']
        oai_ver = config['openai']['version']
        oai_type = config['openai']['type']

        os.environ["OPENAI_API_TYPE"] = oai_type
        os.environ["OPENAI_API_BASE"] = oai_base
        os.environ["OPENAI_API_VERSION"] = oai_ver
        os.environ['OPENAI_API_KEY'] = oai_key

        openai.api_type = oai_type
        openai.api_key = oai_key
        openai.api_base = oai_base
        openai.api_version = oai_ver

        azure_subscription_key = config['azure_form_recognizer']['subscription_key']
        azure_subscription_endpoint = config['azure_form_recognizer']['endpoint']

        acc_name = config['storage_account']['sa_name']
        acc_key = config['storage_account']['sa_key']
        return azure_subscription_key, azure_subscription_endpoint, acc_name, acc_key
    
    except:
        pass



def remove_extension(filename):
    if filename != None:
        return filename.split('.')[0]


def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')



def user_file_upload_to_blob(file_content, file_name, blob_service_client):
    
    # Your Azure Blob Storage container name
    container_name = "Your Azure File Container Name"
    container_client = blob_service_client.get_container_client(container_name)
    blob_name = os.path.basename(file_name)

    try:
        blob_client = container_client.get_blob_client(blob_name)

        # Upload file content to blob
        blob_client.upload_blob(file_content, overwrite=True)
        st.success(f"File '{file_name}' uploaded successfully to Azure Blob Storage.")
        return file_name
    except Exception as e:
        st.error(f"Error uploading file to Azure Blob Storage: {e}")

def download_input_folder_files(home, file_name, blob_service_client, container_name):
    #Get a reference to the container
    container_client = blob_service_client.get_container_client(container_name)
 
    # List blobs in the specified folder
    blobs = container_client.list_blobs()
    print(blobs)
    file_path = ""
    local_docs_path = os.path.join(home, "Trained_Docs")
    os.makedirs(local_docs_path, exist_ok=True)
    # # Iterate through the blobs and download them with their folder structure
    for blob in blobs:
        blob_name = blob.name
        if blob_name == file_name:
            st.success("Found file name from azure")
            print(blob_name)
           
            blob_client = container_client.get_blob_client(blob_name)
            # destination_path = home + "/IndexingFiles"
            destination_path = os.path.join(local_docs_path, f"{blob_name}")
            print(destination_path)

            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            print(os.path.dirname(destination_path))

            with open(destination_path, "wb") as file:
                data = blob_client.download_blob()
                file.write(data.readall())

def download_indexes_folder(container_name, selected_file, home, blob_service_client):
    # selected_file = selected_file 
    container_client = blob_service_client.get_container_client(container_name)

    # List blobs with the specified prefix
    blob_list = container_client.list_blobs(name_starts_with=selected_file)
    #create indexes foler
    local_indexes_path = os.path.join(home, "Indexes")
    os.makedirs(local_indexes_path, exist_ok=True)
    # Download each blob's content
    for blob in blob_list:
        blob_client = container_client.get_blob_client(blob.name)

        # Extract file name from blob name
        file_name = os.path.basename(blob.name)

       # Create local folder inside "Indexes" if not exists
        local_folder_path = os.path.join(local_indexes_path, selected_file)
        os.makedirs(local_folder_path, exist_ok=True)

        # Download blob content to local file
        blob_content = blob_client.download_blob().readall()
        local_file_path = os.path.join(local_folder_path, file_name)
        with open(local_file_path, "wb") as file:
            file.write(blob_content)

        print(f"Downloaded: {blob.name} to {local_folder_path}")



def trigger_function_app(name):
    # Replace 'your_function_app_url' with the actual URL of your Function App
    function_app_url = 'Your Function App URL Name'
        
    # Trigger the URL
    param = {
        "name":name
    }
    response = requests.post(function_app_url, params=param)

    # Check the response status
    if response.status_code == 200:
        print(f"Created Index successfully of file name: {name}")
        print(f"Response: {response.text}")
    else:
        print(f"Failed to trigger Function App. Status code: {response.status_code}")
     
def fetch_key_value_data(home, filename, azure_subscription_key, azure_subscription_endpoint):
    st.success('Please wait...')
    print("You clicked fetch_key button")
    print("inside fetch - Home:", home)
    subfolder_name = "Trained_Docs"

    # Create the full path to the subfolder
    filee = os.path.join(home, subfolder_name)

    # Check if the subfolder exists, and create it if not
    if not os.path.exists(filee):
        os.makedirs(filee)

    answer=''
    try:
        # filename_without_extension = remove_extension(filename)
        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', chunk_size=1)
        ques1="Give me the Summary in 80 words"
        ques =  "What is the Contract Number\
                What is the Document name \
                What is the Contract Name (Document heading) \
                In which  Language\
                What is the buyer Entity Name\
                What is the buyer Party Country\
                What is the buyer Address\
                What is the Contract StartDate\
                What is the Contract EndDate\
                What is the Vendor Entity Name\
                What is the Vendor Address\
                What is the First Party Entity Signature Date\
                                                        "
        
    
        template= """
        Use the following pieces of context to answer the question at the end. Analyze the questions carefully. Most chances there will be answer in the context that I will provide. Go through it, you should be able to find the answer. If you can't find the answer, just give \"N/A\", don't try to make up an answer. Wherever felt necessary answer in points.
        The answer should be specific to the question that has been asked.
        The answer should not be summarized.
        Each step answer will be stored in a dictonary where key will be step and answer will be the value also take the date format same as in the document
        Answer the question only from the context provided.
        NOTE: There might be html tables in the context, interpret it as a table and answer accordingly if the question is from the table.
        # {context}

        # Question: {question}
        # Helpful Answer:"""
        PROMPT = PromptTemplate(template=template,
        input_variables = ['context', 'question'])

        print("File: ",f"{filename}/index.faiss")
        index_folder = os.path.join(home, 'Indexes')
        file_index_path = os.path.join(index_folder, f"{filename}")
        print(file_index_path)
        retriever = FAISS.load_local(file_index_path, embeddings).as_retriever()
        chain_type_kwargs = {"prompt": PROMPT}
        qa= RetrievalQA.from_chain_type(llm=AzureOpenAI(deployment_name='text-davinci-003',model_name='text-davinci-003',temperature=0,max_tokens=1250),chain_type='stuff',retriever=retriever,chain_type_kwargs=chain_type_kwargs)
        answer= qa.run({'query': ques})
        answer1=qa.run({'query': ques1})
        # doc_link_name= filename.replace(" ", "%20")
        ans_lower = answer.lower()
        ans_lower1 = answer1.lower()
        if ans_lower.find("I don't know".lower()) != -1:
            doc_link=""
        cat='directans'

        json_body={"Status Code": "200", "Status": "Success","Cat":cat,"Ans":answer}

        final_result = {
                        "Cat":cat,
                        "Question": ques,
                        "Ans":answer,                          
                    }
        final_result1={
                        "Cat":cat,
                        "Question": ques1,
                        "Ans":answer1,     
        }
        for k,v in final_result.items():
            print(k, " : ", v)
        workstatus=1
        print(answer)
        print(answer1)
        
        lines = answer.strip().split('\n')

        # Create a dictionary from the lines
        result_dict = {}
        for line in lines:
            if ':' in line:
                key, value = map(str.strip, line.split(':', 1))
                result_dict[key] = value
            else:
    # Handle cases where the line doesn't contain the ':' character
                pass 
        

        # Print the resulting dictionary
        print(result_dict)
        df = pd.DataFrame(list(result_dict.items()), columns=['Topic', 'Answer'], index=range(1, len(result_dict) + 1))
        st.write("**Summary:**" + "<br>" + str(answer1), unsafe_allow_html=True)
        st.table(df)

        # Convert DataFrame to CSV and cache it
        csv_data = convert_df_to_csv(df)
        # return sum, csv_data

        # Display the download button
        st.download_button(
        label="Download As CSV",
        data=csv_data,
        file_name='downloaded_data.csv',
        mime='text/csv',
        )
    except:
        print('Exception: {}'.format(traceback.format_exc()))
    # except:
    #     print('Exception: {}'.format(traceback.format_exc()))


def user_answer(home, user_ques, filename):
    answer=''  
    try:
        
        # filename_without_extension = remove_extension(filename)
        # file_index_path = os.path.join(home, filename_without_extension)
        index_folder = os.path.join(home, 'Indexes')
        file_index_path = os.path.join(index_folder, f"{filename}")
        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', chunk_size=1)
        ques = user_ques
    
        template= """
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say \"I don't know\", don't try to make up an answer.
        Wherever felt necessary answer in points.
        The answer should be specific to the question that has been asked.
        The answer should not be summarized.
        Each step answer will be stored in a dictonary where key will be step and answer will be the value also take the date format same as in the document
        Answer the question only from the context provided.
        NOTE: There might be html tables in the context, interpret it as a table and answer accordingly if the question is from the table.
        # {context}

        # Question: {question}
        # Helpful Answer:"""
        PROMPT = PromptTemplate(template=template,
        input_variables = ['context', 'question'])

        print("else",f"{filename}/index.faiss")

        print(file_index_path)
        retriever = FAISS.load_local(file_index_path, embeddings).as_retriever()
        chain_type_kwargs = {"prompt": PROMPT}
        qa= RetrievalQA.from_chain_type(llm=AzureOpenAI(deployment_name='text-davinci-003',model_name='text-davinci-003',temperature=0,max_tokens=1250),chain_type='stuff',retriever=retriever,chain_type_kwargs=chain_type_kwargs)
        answer= qa.run({'query': ques})
        
        # doc_link_name= filename_without_extension.replace(" ", "%20")
        ans_lower = answer.lower()
        cat='directans'
        final_result = {
                        "Cat":cat,
                        "Question": ques,
                        "Ans":answer,                          
                    }
        
        for k,v in final_result.items():
            print(k, " : ", v)
            
        return answer
        # st.write(answer)
        
    except:
        print('Exception: {}'.format(traceback.format_exc()))


# def get_blob_files(blob_service_client, container_name):
#     container_client = blob_service_client.get_container_client(container_name)
#     blob_list = container_client.list_blobs()
#     return [blob.name for blob in blob_list]

def get_blob_files(blob_service_client, container_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs()
    folder_set = set()

    for blob in blob_list:
        # Extract folder name from blob name
        folder_name = blob.name.split('/')[0]

        # Add the folder name to the set
        folder_set.add(folder_name)

    # Convert set to a list
    folder_list = list(folder_set)

    return folder_list
    # files = [blob.name for blob in blob_list]
    # return files


# Register the delete_folders function to be called when the program exits
def delete_dir(home):
    print("inside deletion function")
    docs_folder = "Trained_Docs"
    indexes_folder = "Indexes"
    docs = os.path.join(home, docs_folder)
    index = os.path.join(home, indexes_folder)
    
    try:
        # Attempt to remove the directory and its contents
        if os.path.exists(docs):
            shutil.rmtree(docs)
            print(f"Directory '{docs}' removed successfully.")

        if os.path.exists(index):
            shutil.rmtree(index)
            print(f"Directory '{index}' removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{docs,index}' not found.")
    except OSError as e:
        print(f"Error removing directory: {e}")

user_filecontent = ""
user_filename = ""

def main():
    logging.info('Function triggered with CWD: {}'.format(os.getcwd()))
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser('~')
    # if "HOME" in os.environ:
    #     home = os.environ["HOME"]
    # home = os.path.join('/')
    logging.info("Home Path: {home}")
    if "deletion_done" not in st.session_state:
        delete_dir(home)
        st.session_state.deletion_done = True

    azure_subscription_key, azure_subscription_endpoint, acc_name,acc_key=config_setting(ROOT_DIR,home)

    sa_conn_string = f"DefaultEndpointsProtocol=https;AccountName={acc_name};AccountKey={acc_key};EndpointSuffix=core.windows.net"
    blob_service_client  = BlobServiceClient.from_connection_string(conn_str = sa_conn_string)

    index_container_name = f"Index Azure COntainer Name"
    file_container_name= f"File Azure COntainer Name"

    logging.info("Config file read")
    tokenizer = nltk.data.load(os.path.join('english.pickle'))

    st.title("Contract Information Extractor  🤖")    
    selected_file = st.sidebar.selectbox("Choose a file",get_blob_files(blob_service_client, index_container_name), index = None, placeholder="Choose an option" )
    user_selected_filename = remove_extension(selected_file)
    if selected_file:
        st.success(f"Selected file: {user_selected_filename}") 
        download_indexes_folder(index_container_name, selected_file, home, blob_service_client)    
        download_input_folder_files(home, selected_file, blob_service_client, file_container_name)  
        get_result=st.sidebar.button("Fetch Key Value Pairs")
                                # type="primary")   
        if get_result:
            st.success("Fetching Key Value Pairs...")
            fetch_key_value_data(home, selected_file, azure_subscription_key, azure_subscription_endpoint) # Ensure this function returns a value
    
    # blob_index_path(selected_file, blob_service_client, index_container_name)
    st.sidebar.subheader("If your file is not in the drop down please upload below")
    uploaded_file = st.sidebar.file_uploader("Choose a file",type=["pdf"])
    if uploaded_file is not None:
        # To read file as bytes:
        file_path = os.path.join("Docs", uploaded_file.name)
        global user_filecontent
        global user_filename
        user_filecontent = uploaded_file.read()
        user_filename = uploaded_file.name
        st.success(user_filename)
        
        upload=st.sidebar.button("Upload")
                                    # type="primary")
        
        # Blank text area will present all the time
        if upload:
            ab = user_file_upload_to_blob(user_filecontent, user_filename, blob_service_client)
            print("uploaded to blob {ab}")
            trigger_function_app(user_filename)
            st.rerun()
    if selected_file == None:
        st.header("Please choose a file from the drop down menu")
        

    # Initialize chat history
    # if selected_file:
    # qna_button = st.sidebar.button("QnA",type='primary')
    # qna_button = st.button("QnA")
    if selected_file != None:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input    
        # if qna_button:
        if prompt := st.chat_input("Ask your query"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = user_answer(home, prompt, selected_file)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()


