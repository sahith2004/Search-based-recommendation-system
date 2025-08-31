from dotenv import load_dotenv
load_dotenv()
import boto3
##### LLAMAPARSE #####
from llama_parse import LlamaParse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore

os.environ["PINECONE_API_KEY"] = "81049159-b6de-4f54-a9fa-6179124b5335"
os.environ["PINECONE_INDEX_NAME"] = "quickstart"
llamaparse_api_key = "llx-AAAGYvrbYDxve5HwUdOfbaM100rplsw7q1kYmW4OAFjmuEmT"

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
def create_vector_database():

   
    
   
    
    loader = DirectoryLoader('Factsheet-Markdown', glob="**/*.md", show_progress=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    
    len(docs)
    
    index = PineconeVectorStore.from_documents(docs, bedrock_embeddings, index_name="quickstart")

    print('Vector DB created successfully !')


if __name__ == "__main__":
    create_vector_database()