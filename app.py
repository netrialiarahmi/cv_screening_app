import streamlit as st
import pandas as pd
from modules.extractor import extract_text_from_pdf
from modules.scorer import score_with_llama
from modules.utils import save_results

st.set_page_config(page_title="AI CV Screening System", layout="wide")

# --- Custom Navigation Bar ---
st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        background-color: #F4F6FF;
        padding: 10px 0;
        border-radius: 10px;
        margin-bottom: 25px;
    }
    .nav-item {
        font-weight: 600;
        font-size: 16px;
        color: #333;
        text-decoration: none;
        padding: 10px 25px;
        border-radius: 8px;
        transition: 0.3s;
    }
    .nav-item:hover {
        background-color: #007BFF;
        color: white;
    }
    .active {
        background-color: #007BFF;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Navigation State ---
if "page" not in st.session_state:
    st.session_state.page = "Upload"

col1, col2 = st.columns([1,1])

with col1:
    if st.button("📤 Upload & Screening", use_container_width=True):
        st.session_state.page = "Upload"

with col2:
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"

# --- Page 1: Upload & Screening ---
if st.session_state.page == "Upload":
    st.title("📤 Upload CVs & Job Description")

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
                progress.progress((i+1)/len(uploaded_files))
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

# --- Page 2: Dashboard ---
elif st.session_state.page == "Dashboard":
    st.title("📊 Screening Dashboard")

    if "results" in st.session_state:
        st.dataframe(st.session_state["results"], use_container_width=True)

        csv = st.session_state["results"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=csv,
            file_name="cv_screening_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("⚠️ No results yet. Please run screening first.")
