import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text
import tempfile
from random import sample

try:
    summarizer = pipeline("summarization")
    qa_pipeline = pipeline("question-answering")
except Exception as e:
    st.error("❌ Failed to load Hugging Face models. Please check your internet connection.")
    st.stop()


def extract_content(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                text = extract_text(tmp_file.name)
        else:
            text = uploaded_file.read().decode("utf-8")
        return text
    except Exception as e:
        st.error("❌ Could not read the uploaded file.")
        st.exception(e)
        return ""

st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("🤖 Smart Assistant for Research Summarization")
st.markdown("Upload your document, ask questions, and challenge your understanding.")


uploaded_file = st.file_uploader("📎 Upload a PDF or TXT document", type=["pdf", "txt"])

if uploaded_file:
    text = extract_content(uploaded_file)

    if text:
        
        st.subheader("📌 Auto Summary")
        try:
            summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
            st.success(summary)
        except Exception as e:
            st.error("❌ Failed to generate summary.")
            st.exception(e)

        mode = st.radio("🧭 Choose Mode", ["Ask Anything", "Challenge Me"])
        if mode == "Ask Anything":
            question = st.text_input("❓ Ask a question based on the document:")
            if question:
                try:
                    answer = qa_pipeline(question=question, context=text)
                    st.write("🧠 **Answer:**", answer["answer"])
                    st.caption("📚 Based on the uploaded document.")
                except Exception as e:
                    st.error("❌ Failed to generate an answer.")
                    st.exception(e)
                    
        elif mode == "Challenge Me":
            st.subheader("🎯 Logic-Based Questions")
            questions = sample([
                "What is the main purpose of the document?",
                "What is one key insight presented?",
                "Summarize a major argument or finding."
            ], 3)

            for idx, q in enumerate(questions):
                user_answer = st.text_input(f"📝 Q{idx+1}: {q}")
                if user_answer:
                    try:
                        model_answer = qa_pipeline(question=q, context=text)["answer"]
                        if user_answer.lower() in model_answer.lower():
                            st.success("✅ Correct!")
                        else:
                            st.warning(f"❌ Correct Answer: {model_answer}")
                    except Exception as e:
                        st.error("❌ Failed to evaluate the answer.")
                        st.exception(e)
