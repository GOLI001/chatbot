from django.shortcuts import render
from django.http import JsonResponse
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from getpass import getpass
import os

# Set up Hugging Face token
token = getpass("Enter your Hugging Face Hub token: ")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Load language model
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl")

# Load documents
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=Ld3iHVlvhIM")
documents = loader.load()

# Create the vector store
db = FAISS.from_documents(documents, embeddings)

# Create the retriever
retriever = db.as_retriever()

# Create the QA model
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

def get_response(request):
    if request.method == 'POST':
        query = request.POST.get('query', '')
        result = qa({"query": query})
        response = result['result']
        return JsonResponse({'response': response})
    return render(request, 'index.html')
