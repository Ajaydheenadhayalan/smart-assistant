import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text
import tempfile
import google.generativeai as genai

# Load Hugging Face models only once
if "qa_model" not in st.session_state or "summarizer" not in st.session_state:
    try:
        st.session_state.qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        st.session_state.summarizer = pipeline("summarization")
    except Exception as e:
        st.error("Failed to load NLP models.")
        st.stop()

# Configure Gemini
genai.configure(api_key=st.secrets["gemini_api_key"])

# Extract text from uploaded document
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
        st.error("Could not read the uploaded file.")
        st.exception(e)
        return ""

# Generate questions with Gemini
def generate_questions_with_gemini(text: str):
    prompt = f"""
    Read the following document and generate three logic-based or comprehension-testing questions.

    \"\"\"{text[:4000]}\"\"\"

    Format:
    1. ...
    2. ...
    3. ...
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        lines = response.text.split("\n")
        questions = [q.strip("1234567890. ").strip() for q in lines if q.strip()]
        return questions[:3]
    except Exception as e:
        st.error("Gemini failed to generate questions.")
        st.exception(e)
        return []

# Streamlit App
st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("Smart Assistant for Research Summarization")
st.markdown("Upload your document, ask questions, and challenge your understanding.")

uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

if uploaded_file:
    text = extract_content(uploaded_file)

    if text:
        st.subheader("Auto Summary")
        try:
            summary = st.session_state.summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
            st.success(summary)
        except Exception as e:
            st.error("Failed to generate summary.")
            st.exception(e)

        mode = st.radio("Choose Mode", ["Ask Anything", "Challenge Me"])

        # Ask Anything Mode
        if mode == "Ask Anything":
            question = st.text_input("Ask a question based on the document:")
            if question:
                try:
                    answer = st.session_state.qa_model(question=question, context=text)
                    st.markdown(f"**Answer:** {answer['answer']}")
                    st.caption("Based on uploaded document.")
                except Exception as e:
                    st.error("Failed to answer the question.")
                    st.exception(e)

        # Challenge Me Mode
        elif mode == "Challenge Me":
            st.subheader("Comprehension Challenge")

            if "stored_questions" not in st.session_state:
                st.session_state.stored_questions = generate_questions_with_gemini(text)

            questions = st.session_state.stored_questions

            if "user_answers" not in st.session_state:
                st.session_state.user_answers = [""] * len(questions)

            for idx, question in enumerate(questions):
                st.markdown(f"Q{idx+1}: {question}")
                st.session_state.user_answers[idx] = st.text_area(
                    f"Your Answer to Q{idx+1}",
                    value=st.session_state.user_answers[idx],
                    key=f"user_answer_{idx}"
                )

            if st.button("Evaluate Answers"):
                st.subheader("Evaluation Results")
                correct_count = 0

                for idx, (question, user_answer) in enumerate(zip(questions, st.session_state.user_answers)):
                    if not user_answer.strip():
                        st.warning(f"Q{idx+1}: No answer provided.")
                        continue
                    try:
                        model_answer = st.session_state.qa_model(question=question, context=text)["answer"]
                        if user_answer.strip().lower() in model_answer.lower():
                            st.success(f"Q{idx+1}: Correct!")
                            correct_count += 1
                        else:
                            st.error(f"Q{idx+1}: Incorrect.")
                            with st.expander("View Model's Answer"):
                                st.info(model_answer)
                    except Exception as e:
                        st.error(f"Error evaluating Q{idx+1}")
                        st.exception(e)

                st.markdown(f"**Final Score: {correct_count} / {len(questions)} correct**")
