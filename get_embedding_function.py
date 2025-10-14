from langchain_community.embeddings import OllamaEmbeddings  # Updated
from langchain_ollama import OllamaEmbeddings
#import google.generativeai as genai
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

#os.environ["GOOGLE_API_KEY"] = "Put your Google API key here if you want to use their embedding model"


#genai.configure(api_key="Put your Google API key here if you want to use their embedding model")

"""
def get_embedding_function():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

"""

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text:v1.5")


