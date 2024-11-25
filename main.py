from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os

from dotenv import load_dotenv
load_dotenv()

def main():
    docs = load_docs()
    chunks = split_docs(docs)
    add_chunks_to_chroma(chunks)


def load_docs():
    document_loader = PyPDFDirectoryLoader(os.getenv("DATA_PATH"))
    return document_loader.load()


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80, length_function=len, is_separator_regex=False)
    return splitter.split_documents(docs)


def add_chunks_to_chroma(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=os.getenv("CHROMA_PATH"), embedding_function=embeddings)
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)


def calculate_chunk_ids(chunks):
    # create IDs with schema "Page_Source:Page_Number:Chunk_Index", e.g. "data/monopoly.pdf:6:2"
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

if __name__ == "__main__":
    main()

