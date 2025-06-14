import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb import HttpClient
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = "ncert_data/class10"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN")

# Initialize Chroma Cloud client
client = HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=Settings(anonymized_telemetry=False),
    headers={"Authorization": f"Bearer {CHROMA_TOKEN}"} if CHROMA_TOKEN else {}
)

def ingest_subject_pdfs(subject_path, subject_name):
    print(f"Processing subject: {subject_name} from {subject_path}")

    all_documents = []

    # Recursively load PDFs from all chapters inside this subject folder
    for root, _, files in os.walk(subject_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Loading {pdf_path}...")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                # Optionally split large docs into chunks for better embedding
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(docs)
                all_documents.extend(split_docs)

    if not all_documents:
        print(f"No documents found for subject {subject_name}, skipping ingestion.")
        return

    # Create embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Uploading to Chroma Cloud collection: {subject_name} ...")

    vectordb = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        collection_name=subject_name,
        client=client
    )

    print(f"✅ Ingestion complete for subject: {subject_name}\n")

def main():
    # List of subjects expected in class10 folder
    subjects = ["science", "maths", "history", "geography", "politicalscience", "economics"]

    for subject in subjects:
        subject_folder = os.path.join(BASE_DIR, subject)
        if os.path.exists(subject_folder):
            ingest_subject_pdfs(subject_folder, subject)
        else:
            print(f"Subject folder does not exist: {subject_folder}")

if __name__ == "__main__":
    main()
