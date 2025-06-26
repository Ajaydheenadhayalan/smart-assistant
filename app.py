import streamlit as st
import tempfile
from pdfminer.high_level import extract_text
from transformers import pipeline, Pipeline
from random import sample

st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("📄 Smart Assistant for Research Summarization")

# Try loading NLP pipelines with error handling
try:
    summarizer: Pipeline = pipeline("summarization")
    qa_pipeline: Pipeline = pipeline("question-answering")
    model_loaded = True
except Exception as e:
    st.error("❌ Could not connect to Hugging Face to load models. Check your internet connection.")
    st.exception(e)
    model_loaded = False

# PDF/TXT content extraction
def extract_content(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            text = extract_text(tmp_file.name)
    else:
        text = uploaded_file.read().decode("utf-8")
    return text

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    text = extract_content(uploaded_file)

    if model_loaded:
        st.subheader("📌 Document Summary")
        with st.spinner("Summarizing..."):
            try:
                summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
                st.success(summary)
            except Exception as e:
                st.error("Summary generation failed.")
                st.exception(e)

        # Select Mode
        mode = st.radio("Select Mode", ["Ask Anything", "Challenge Me"])

        if mode == "Ask Anything":
            question = st.text_input("Ask a question based on the document:")
            if question:
                try:
                    answer = qa_pipeline(question=question, context=text)
                    st.write("🧠 **Answer:**", answer["answer"])
                    st.caption("📚 Justification: Based on document context.")
                except Exception as e:
                    st.error("Could not answer the question.")
                    st.exception(e)

        elif mode == "Challenge Me":
            st.subheader("🧠 Challenge Questions")
            base_questions = [
                "What is the main purpose of the document?",
                "What is one key insight presented?",
                "Summarize a major argument or finding."
            ]
            questions = sample(base_questions, 3)

            for idx, q in enumerate(questions):
                user_answer = st.text_input(f"Q{idx+1}: {q}")
                if user_answer:
                    try:
                        model_answer = qa_pipeline(question=q, context=text)["answer"]
                        if user_answer.lower() in model_answer.lower():
                            st.success("✅ Correct!")
                        else:
                            st.warning(f"❌ Incorrect. Correct: {model_answer}")
                    except Exception as e:
                        st.error("Answer evaluation failed.")
                        st.exception(e)
    else:
        st.warning("Document uploaded, but NLP models are not loaded due to network issue.")
