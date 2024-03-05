import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PDF_PATH = "data/iswd-lec1.pdf"
EMBEDDINGS_MODEL = "paraphrase-MiniLM-L6-v2"
PROMPT_KEYWORD = "PROMETHEE"
VECTORSTORE_PATH = "chroma_db_open_source"
LLM_MODEL = "google/flan-t5-large"


def fill_vectorstore(pdf_path=PDF_PATH):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separator="\n"
    )
    docs = text_splitter.split_documents(documents)
    # embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)
    db = Chroma.from_documents(docs, embeddings, persist_directory=VECTORSTORE_PATH)
    db.persist()


def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL)
    return Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)


def load_llm():
    return HuggingFaceHub(
        repo_id=LLM_MODEL, model_kwargs={"temperature": 0, "max_length": 64}
    )


def generate_chain(model, db):
    return RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", retriever=db.as_retriever()
    )


def generate_prompt(keyword=PROMPT_KEYWORD):
    return f"Explain shortly {keyword}"


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    llm = load_llm()
    chain = generate_chain(llm, vectorstore)

    # collection = vectorstore.get()
    # print(vectorstore.similarity_search(query, k=5))

    prompt = generate_prompt()
    response = chain(prompt)
    print(response)
