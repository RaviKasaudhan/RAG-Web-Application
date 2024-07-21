import logging
import azure.functions as func
import azure.functions as func
import html, json, logging, openai, os, nltk.data, time, traceback
# from azure.functions import HttpRequest, HttpResponse
from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
# import requests
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import shutil
# import urllib.parse

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

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


def get_document_text(filepath, azure_subscription_key, azure_subscription_endpoint):
    offset = 0
    page_map = []
    form_recognizer_client = DocumentAnalysisClient(endpoint=azure_subscription_endpoint, credential=(AzureKeyCredential(azure_subscription_key)), headers={"x-ms-useragent": "azure-search-chat-demo/1.0.0"})
    with open(filepath, "rb") as f:
        poller = form_recognizer_client.begin_analyze_document("prebuilt-layout", document = f)
    form_recognizer_results = poller.result()

    for page_num, page in enumerate(form_recognizer_results.pages):
        tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

        # mark all positions of the table spans in the page
        page_offset = page.spans[0].offset
        page_length = page.spans[0].length
        table_chars = [-1]*page_length
        for table_id, table in enumerate(tables_on_page):
            for span in table.spans:
                # replace all table spans with "table_id" in table_chars array
                for i in range(span.length):
                    idx = span.offset - page_offset + i
                    if idx >=0 and idx < page_length:
                        table_chars[idx] = table_id
        
        page_text = ""
        added_tables = set()
        for idx, table_id in enumerate(table_chars):
            if table_id == -1:
                page_text += form_recognizer_results.content[page_offset + idx]
            elif not table_id in added_tables:
                page_text += table_to_html(tables_on_page[table_id])
                added_tables.add(table_id)

        page_text += " "
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)

    return page_map


def split_text(page_map):
    MAX_SECTION_LENGTH = 1000
    SENTENCE_SEARCH_LIMIT = 100
    SECTION_OVERLAP = 100

    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

    def find_page(offset):
        l = len(page_map)
        for i in range(l - 1):
            if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
                return i
        return l - 1

    all_text = "".join(p[2] for p in page_map)
    length = len(all_text)
    start = 0
    end = length
    while start + SECTION_OVERLAP < length:
        last_word = -1
        end = start + MAX_SECTION_LENGTH

        if end > length:
            end = length
        else:
            # Try to find the end of the sentence
            while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
                if all_text[end] in WORDS_BREAKS:
                    last_word = end
                end += 1
            if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
                end = last_word # Fall back to at least keeping a whole word
        if end < length:
            end += 1

        # Try to find the start of the sentence or at least a whole word boundary
        last_word = -1
        while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
            if all_text[start] in WORDS_BREAKS:
                last_word = start
            start -= 1
        if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
            start = last_word
        if start > 0:
            start += 1

        section_text = all_text[start:end]
        yield (section_text, find_page(start))

        last_table_start = section_text.rfind("<table")
        if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
            # If the section ends with an unclosed table, we need to start the next section with the table.
            # If table starts inside SENTENCE_SEARCH_LIMIT, we ignore it, as that will cause an infinite loop for tables longer than MAX_SECTION_LENGTH
            # If last table starts inside SECTION_OVERLAP, keep overlapping
#             print(f"Section ends with unclosed table, starting next section with the table at page {find_page(start)} offset {start} table start {last_table_start}")
            start = min(end - SECTION_OVERLAP, start + last_table_start)
        else:
            start = end - SECTION_OVERLAP
        
    if start + SECTION_OVERLAP < end:
        yield (all_text[start:end], find_page(start))


def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html


def create_sections(filename, page_map):
    documents = []
    file_name = os.path.basename(filename)
    for i, (content, pagenum) in enumerate(split_text(page_map)):
        section = {
            "source": file_name
        }
        documents.append(Document(page_content = content, metadata = section))
        # if use_vectors:
        #     section["embedding"] = compute_embedding(content)
    return documents


def index_file_upload_to_blob(container_name, filename, file_index_path, blob_service_client):
    # Get a reference to the container
    logging.info("index file upload function activated")
    index_container_client = blob_service_client.get_container_client(container_name)

    with open(file=os.path.join(file_index_path,"index.faiss"), mode="rb") as data:
        blob_client = index_container_client.upload_blob(name=f"{filename}/index.faiss", data=data, overwrite=True)
        
    with open(file=os.path.join(file_index_path,"index.pkl"), mode="rb") as data:
        blob_client = index_container_client.upload_blob(name=f"{filename}/index.pkl", data=data, overwrite=True)
    logging.info("index file uploaded to azure blob")
    print(f"Files uploaded successfully:")

def download_input_folder_files(home, file_name, blob_service_client, container_name):
    logging.info("inside function app download user file function")
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
                logging.info("user file is downloaded via the download function")
    

## Creating Index
def create_index(home, filename, index_container_name, blob_service_client, azure_subscription_key, azure_subscription_endpoint):
    # Define the subfolder name
    logging.info("create_index function activated")
    subfolder_name = "Trained_Docs"

    # Create the full path to the subfolder
    filee = os.path.join(home, subfolder_name)

    # Check if the subfolder exists, and create it if not
    if not os.path.exists(filee):
        os.makedirs(filee)
    logging.info(f"{filee} Created")
    try:
        print('Home Path: {}'.format(home))
        print("Config file read")
        # tokenizer = nltk.data.load('./english.pickle')
        
        index_folder = os.path.join(home, 'Indexes')

        if not os.path.isdir(index_folder):
            os.mkdir(index_folder)

        try:
            file_path = os.path.join(filee, f"{filename}")
            file_index_path = os.path.join(index_folder, f"{filename}")
            page_map = get_document_text(file_path, azure_subscription_key, azure_subscription_endpoint)
            docs = create_sections(os.path.basename(filename), page_map)
            embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', chunk_size=1)
            db = FAISS.from_documents(docs, embeddings)
            db.save_local(file_index_path)
            index_file_upload_to_blob(index_container_name, filename, file_index_path, blob_service_client)

            # time_taken = (time.time() - start_time)
            print('>> ',  '-' , filename, '- Success')
            time.sleep(10)
            logging.info("last line of create_index function")
        
        except:
            return 'Exception: {}'.format(traceback.format_exc())
            print('Exception: {}'.format(traceback.format_exc()))
    except:
        return 'Exception: {}'.format(traceback.format_exc())
        print('Exception: {}'.format(traceback.format_exc()))

def delete_dir(home):
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


def test(name):
    logging.info('Function triggered with CWD: {}'.format(os.getcwd()))
    logging.info("inside test function..started")
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    home = os.path.expanduser('~')
    logging.info(home)
  #  delete_dir(home)
    azure_subscription_key, azure_subscription_endpoint, acc_name,acc_key=config_setting(ROOT_DIR,home)

    sa_conn_string = f"DefaultEndpointsProtocol=https;AccountName={acc_name};AccountKey={acc_key};EndpointSuffix=core.windows.net"
    blob_service_client  = BlobServiceClient.from_connection_string(conn_str = sa_conn_string)

    index_container_name = f"Index Azure COntainer Name"
    file_container_name= f"File Azure COntainer Name"
    download_input_folder_files(home, name, blob_service_client, file_container_name)
    create_index(home, name, index_container_name, blob_service_client, azure_subscription_key, azure_subscription_endpoint)
    logging.info("test function executed")
    # return "Index generation function end"


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
   
    name = req.params.get('name')
    logging.info(f"filename we got as a parameter to function app url is {name}")
   
    if name:
        logging.info("now activating test function via main function")
        test(name)
        
        return func.HttpResponse(f"Hello. This HTTP triggered function executed successfully.")
    else:
        logging.info("Inside else block")
        return func.HttpResponse(
             status_code=200
        )


    