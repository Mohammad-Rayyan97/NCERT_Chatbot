import streamlit as st
from main import detect_subject_from_query, create_qa_chain

st.set_page_config(page_title="NCERT Assistant", layout="wide")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# Sidebar
with st.sidebar:
    st.title("📘 NCERT Ray-bot for My Dear Angels")
    st.markdown("Ask questions from NCERT Class 10 subjects like:")
    st.markdown("- Science")
    st.markdown("- Maths")
    st.markdown("- History")
    st.markdown("- Geography")
    st.markdown("- Civics")
    st.markdown("- Economics")
    st.markdown("Built with 🧠 LangChain + Gemma2-9B-IT")    
    st.markdown("### 💡 How to Ask a Good Question")

    # Query Guidelines
    st.markdown("""
**🧠 Before asking a question, make sure your query is clear and complete.**

### ✅ Good Query Examples:
- _"Explain me **photosynthesis** from **science**."_  <br>
- _"Give me **2 MCQs** from chapter **Chemical Reactions and Equations**."_  <br>
- _"Explain me **role of Gandhi** in **National Movement** in **history**."_  <br>
- _"Give me a brief on **Non-Renewable Sources** in **geography**."_  

### ❌ Poor Query Example:
- _"Explain **nationalism**."_  

<small>➡️ Always mention the subject + chapter or topic for the best answers.</small>

---  
""", unsafe_allow_html=True)



# Chat title
st.title("🧑‍🎓 Ask NCERT Bot")

# Chat history display
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.markdown(chat["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(chat["content"])

# Input area
user_input = st.chat_input("Ask a question from any chapter...")

# On user query
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            subject = detect_subject_from_query(user_input)
            if not subject:
                response = "⚠️ Please include a subject like 'science', 'maths', 'history', etc. in your question."
            else:
                qa_chain = create_qa_chain(subject)
                if qa_chain is None:
                    response = f"❌ No vector DB found for subject `{subject}`."
                else:
                    try:
                        result = qa_chain.invoke(user_input)
                        response = result.get("result", "⚠️ No answer found.")
                    except Exception as e:
                        response = f"❌ Error: {e}"

            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
