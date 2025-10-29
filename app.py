import streamlit as st
import pandas as pd
from modules.extractor import extract_text_from_pdf
from modules.scorer import score_with_llama
from modules.utils import save_results

# --- Streamlit setup ---
st.set_page_config(page_title="AI CV Screening System", layout="wide")

# --- Navigation Bar ---
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        color: #003366;
    }
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 25px;
        margin-bottom: 30px;
    }
    .nav-button {
        background-color: #f5f7ff;
        border: 2px solid #c9d4f0;
        color: #003366;
        font-weight: 600;
        border-radius: 10px;
        padding: 10px 30px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .nav-button:hover {
        background-color: #003366;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>📄 AI CV Screening System</h1>", unsafe_allow_html=True)

# Navigation state
if "page" not in st.session_state:
    st.session_state.page = "Upload"

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("📤 Upload & Screening", use_container_width=True):
        st.session_state.page = "Upload"
with col2:
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"

# --- PAGE 1: UPLOAD & SCREENING ---
if st.session_state.page == "Upload":
    st.subheader("📤 Upload CVs & Input Job Description")

    uploaded_files = st.file_uploader(
        "Upload CVs (PDF files only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    job_description = st.text_area(
        "📝 Job Description",
        placeholder="Paste the job description here...",
        height=200
    )

    if uploaded_files and job_description.strip():
        if st.button("🚀 Run Screening"):
            results = []
            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files))
                text = extract_text_from_pdf(file)
                score, summary = score_with_llama(text, job_description)

                results.append({
                    "Filename": file.name,
                    "Match Score": score,
                    "AI Summary": summary
                })

            df = pd.DataFrame(results)
            st.session_state["results"] = df
            save_results(df)
            st.success("✅ Screening completed successfully!")

    else:
        st.info("Please upload at least one CV and fill in the Job Description.")

# --- PAGE 2: DASHBOARD ---
elif st.session_state.page == "Dashboard":
    st.subheader("📊 Screening Dashboard")

    if "results" in st.session_state:
        df = st.session_state["results"]

        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Filename")["Match Score"])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=csv,
            file_name="cv_screening_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("⚠️ No results yet. Please run screening first.")
