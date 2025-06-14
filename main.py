import json
import os
import re
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from chromadb import HttpClient
from chromadb.config import Settings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage

# Load environment variables
load_dotenv()

# === Configuration ===
CHAPTER_METADATA_PATH = "chapter_metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "deepseek-r1-distill-llama-70b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "443"))
CHROMA_TOKEN = os.getenv("CHROMA_TOKEN")

# === Load Metadata ===
with open(CHAPTER_METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

available_subjects = set(ch["subject"].lower() for ch in metadata.get("class10", []))

SUBJECT_KEYWORDS = {
    "science": ["photosynthesis", "acid", "atom", "electricity", "magnet", "carbon", "reproduction", "light", "chemical", "life processes", "current", "metal", "non-metal"],
    "maths": ["triangle", "trigonometry", "arithmetic", "circle", "coordinate", "algebra", "equation", "polynomial", "mensuration", "statistics", "geometry", "area", "volume"],
    "history": ["nationalism", "gandhi", "congress", "british", "league", "partition", "movement", "colonial", "civil disobedience", "india", "freedom", "struggle"],
    "geography": ["resource", "agriculture", "minerals", "industry", "manufacturing", "soil", "climate", "map", "irrigation", "water", "natural", "energy", "topography"],
    "politicalscience": ["democracy", "constitution", "government", "rights", "election", "federalism", "judiciary", "political", "parliament", "citizen", "law", "power", "party"],
    "economics": ["development", "sectors", "economy", "employment", "GDP", "money", "credit", "trade", "globalization", "consumer", "income", "poverty"],
}

def detect_subject_from_query(query: str):
    query_lower = query.lower()
    for subject in available_subjects:
        if subject in query_lower:
            return subject
    for subject, keywords in SUBJECT_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            return subject
    return None

def find_chapter_info(query: str, subject: str):
    chapter_num_match = re.search(r"chapter\s*(\d+)", query.lower())
    for chapter in metadata.get("class10", []):
        if chapter.get("subject", "").lower() == subject:
            if chapter_num_match and chapter.get("chapter", "") == chapter_num_match.group(1):
                return chapter.get("title", "N/A"), chapter.get("chapter", "N/A")
            elif chapter.get("title", "").lower() in query.lower():
                return chapter.get("title", "N/A"), chapter.get("chapter", "N/A")
    return None, None

def create_qa_chain(subject: str):
    client = HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False)
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(
        client=client,
        collection_name=subject,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGroq(model_name=GROQ_MODEL_NAME, temperature=0.6)

    prompt_template = (
        "You are a helpful educational assistant. Use the following NCERT content to answer the student's question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    def qa_chain_func(inputs):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        inputs.setdefault("context", "Based on NCERT syllabus.")
        return qa.invoke(inputs)

    return qa_chain_func

def clean_llm_output(text: str) -> str:
    content = text.content if isinstance(text, AIMessage) else text
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

def create_quiz_chain():
    llm = ChatGroq(model_name=GROQ_MODEL_NAME, temperature=0.7)

    def quiz_chain_func(inputs):
        num_mcqs_match = re.search(r"(\d+)\s*(mcq|quiz|questions?)", inputs["query"].lower())
        num_mcqs = int(num_mcqs_match.group(1)) if num_mcqs_match else 5

        prompt = (
            f"You are a helpful educational assistant.\n"
            f"Generate a {num_mcqs}-question MCQ quiz (with 4 options each) for NCERT Class 10 {inputs['subject']}."
            f"Chapter: {inputs['chapter_name']} (Chapter {inputs['chapter_number']}).\n"
            f"Topic: {inputs['query']}\n"
            f"Return only the quiz without explanations."
        )
        raw_output = llm.invoke(prompt)
        return clean_llm_output(raw_output)

    return quiz_chain_func

def main():
    print("\U0001F4D8 NCERT Chatbot Ready! (Type 'exit' to quit)\n")

    quiz_chain = create_quiz_chain()

    while True:
        query = input("\U0001F9D1‍\U0001F393 Ask a question: ").strip()
        if query.lower() == "exit":
            print("\U0001F44B Goodbye!")
            break

        subject = detect_subject_from_query(query)
        if not subject:
            print("⚠️ Please include a subject like 'science', 'maths', 'history', etc.")
            continue

        chapter_name, chapter_number = find_chapter_info(query, subject)
        if not chapter_name:
            chapter_name, chapter_number = "N/A", "N/A"

        if any(keyword in query.lower() for keyword in ["quiz", "mcq", "multiple choice"]):
            try:
                quiz = quiz_chain({
                    "query": query,
                    "subject": subject,
                    "chapter_name": chapter_name,
                    "chapter_number": chapter_number
                })
                print("\n📝 Here is your quiz:\n", quiz, "\n")
            except Exception as e:
                print(f"❌ Error generating quiz: {e}")
            continue

        qa_chain = create_qa_chain(subject)
        if qa_chain is None:
            continue

        try:
            result = qa_chain({
                "question": query,
                "query": query,
                "context": "Based on NCERT syllabus."
            })
            raw_output = result.get("result", "⚠️ No result found.")
            answer = clean_llm_output(raw_output)
            print("\n\U0001F9E0 Here is your answer:\n", answer, "\n")
        except Exception as e:
            print(f"❌ Error generating response: {e}")

if __name__ == "__main__":
    main()
