from utils.audio_tag_editor import *

from langchain_ollama import OllamaEmbeddings

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever


def init_vector_store(folder_path: str, llm):
    global vector_store
    global retriever
    
    # embeddings = AzureOpenAIEmbeddings(
    #     azure_endpoint="https://ai-593601083ai249546569384.cognitiveservices.azure.com/",
    #     azure_deployment="text-embedding-3-large",
    #     openai_api_version="2024-02-01"
    #     )
    embeddings = OllamaEmbeddings(model="bona/bge-m3-korean")
    vector_store = store_metadata_in_vector_store(folder_path=folder_path, embeddings=embeddings)
    num_vectors = len(vector_store.get()["ids"])
    
    metadata_field_info  = [
        AttributeInfo(name="filepath", description="Audio file name", type="string"),
        AttributeInfo(name="album", description="Album name", type="string"),
        AttributeInfo(name="title", description="Song title", type="string"),
        AttributeInfo(name="artist", description="Artist name", type="string"),
        AttributeInfo(name="genre", description="Genre of the song", type="string"),
        AttributeInfo(name="year", description="Release year", type="string"),
        AttributeInfo(name="track", description="Track number", type="string"),
        AttributeInfo(name="comment", description="Comments or notes", type="string"),
        AttributeInfo(name="album_artist", description="Album artist name", type="string")
    ]
    document_contents = "metadata of audio files"
    
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents=document_contents,
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": num_vectors},
        enable_limit=True,
        verbose=True,
        score_threshold=1.0
    )
    
def init_vector_store_as_content(folder_path: str, llm):
    global vector_store

    # embeddings = AzureOpenAIEmbeddings(
    #     azure_endpoint="https://ai-593601083ai249546569384.cognitiveservices.azure.com/",
    #     azure_deployment="text-embedding-3-large",
    #     openai_api_version="2024-02-01"
    #     )
    # embeddings = OllamaEmbeddings(model="qllama/multilingual-e5-large-instruct")
    
    vector_store, docs = store_page_content_in_vector_store(folder_path=folder_path, embeddings=embeddings)
    
    return docs
    
def get_vector_store():
    return vector_store

def get_retriever():
    return retriever
