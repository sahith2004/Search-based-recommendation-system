import os
from fastapi import FastAPI, Body
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOllama
import boto3
from langchain.llms.bedrock import Bedrock
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import BedrockEmbeddings

from langchain.chains import RetrievalQA

# bring in our GROQ_API_KEY
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
os.environ["PINECONE_API_KEY"] = "81049159-b6de-4f54-a9fa-6179124b5335"
os.environ["PINECONE_INDEX_NAME"] = "quickstart"
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
**Answer:**

* If you can find relevant information in the documents to answer the question, use it to craft a well-supported answer.
* If you cannot find relevant information, state that you don't know the answer and suggest searching for it elsewhere.

**Additionally:**

* You can mention the limitations of your knowledge and the possibility of the answer existing outside the provided context.
* Encourage the user to provide more context if needed for a more precise answer.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock)

#chat_model = ChatGroq(temperature=0, model_name="Llama2-70b-4096")
#chat_model = ChatOllama(model="llama2", request_timeout=30.0)



def retrieval_qa_chain(llm, prompt, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 7}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
    vectorstore = PineconeVectorStore(index_name="quickstart", embedding=embeddings)

   
    qa_prompt=set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa


print('starting server')
@app.post("/chat")
async def chat(message: str = Body(...)):
  
  print(message)
  bot = qa_bot()
  answer = bot({"query": message})
  print("rag answer ,", answer)
  # Call the LLM model with the chat history
  #   response = llm_model.run(c`hat_history)

  # Extract the response from the LLM model output
  # This might vary depending on the specific model you're using
  #   model_response = response[0]["content"]

  # Return the chat response from the LLM model
  return {"message": answer['result']}