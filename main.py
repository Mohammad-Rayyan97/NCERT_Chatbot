import json
import os
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

# === Configuration ===
CHAPTER_METADATA_PATH = "chapter_metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Load metadata ===
with open(CHAPTER_METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Extract available subjects from metadata
available_subjects = set(
    chapter["subject"].lower() for chapter in metadata.get("class10", [])
)

# === Keyword-based subject detection ===
SUBJECT_KEYWORDS = {
    "science": ["photosynthesis", "acid", "atom", "electricity", "magnet", "carbon", "reproduction", "light", "chemical", "life processes", "current", "metal", "non-metal"],
    "maths": ["triangle", "trigonometry", "arithmetic", "circle", "coordinate", "algebra", "equation", "polynomial", "mensuration", "statistics", "geometry", "area", "volume"],
    "history": ["nationalism", "gandhi", "congress", "british", "league", "partition", "movement", "colonial", "civil disobedience", "india", "freedom", "struggle"],
    "geography": ["resource", "agriculture", "minerals", "industry", "manufacturing", "soil", "climate", "map", "irrigation", "water", "natural", "energy", "topography"],
    "civics": ["democracy", "constitution", "government", "rights", "election", "federalism", "judiciary", "political", "parliament", "citizen", "law", "power", "party"],
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

# === QA chain creation ===
def create_qa_chain(subject: str):
    chroma_dir = f"chroma_db_{subject}"
    if not os.path.exists(chroma_dir):
        print(f"❌ Vector DB folder not found for subject '{subject}': {chroma_dir}")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.6)

    prompt_template = (
        "Use the following NCERT content to answer the student's question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# === Quiz generation chain ===
def create_quiz_chain():
    llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.7)
    quiz_prompt_template = (
        "You are a helpful educational assistant to solve doubts and generate questions for practice.\n"
        "Generate a 5-question MCQ quiz (with 4 options each) covering NCERT Class 10 {subject} .\n"
        "Each question must include options a, b, c, d,.\n"
        "Return only the quiz without explanations.\n\n"
        " and if the query is out of context just say query is out of context try again!!"
        "User Request: {query}"
    )
    quiz_prompt = PromptTemplate(input_variables=["query", "subject"], template=quiz_prompt_template)
    return LLMChain(llm=llm, prompt=quiz_prompt)

# === Main chatbot loop ===
def main():
    print("📘 NCERT Chatbot Ready! (Type 'exit' to quit)\n")

    quiz_chain = create_quiz_chain()

    while True:
        query = input("🧑‍🎓 Ask a question: ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        subject = detect_subject_from_query(query)
        if not subject:
            print("⚠️ Please include a subject like 'science', 'maths', 'history', etc. in your question or use subject-specific terms.")
            continue

        if "quiz" in query.lower() or "mcq" in query.lower() or "multiple choice" in query.lower():
            try:
                quiz = quiz_chain.invoke({"query": query, "subject": subject})
                print("\n📝 Here is your quiz:\n", quiz.get("text", "⚠️ Failed to generate quiz."), "\n")
            except Exception as e:
                print(f"❌ Error generating quiz: {e}")
            continue

        qa_chain = create_qa_chain(subject)
        if qa_chain is None:
            continue

        try:
            result = qa_chain.invoke(query)
            print("\n🧠 Here is your answer:\n", result.get("result", "⚠️ No result found."), "\n")
        except Exception as e:
            print(f"❌ Error generating response: {e}")

if __name__ == "__main__":
    main()
