import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import streamlit as st

@st.cache_resource
def get_llama_pipeline():
    """Load LLaMA model once and cache it."""
    model_id = "meta-llama/Llama-3-8b-instruct"  # or any model you have access to
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        use_auth_token=True
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

def score_with_llama(cv_text, job_description):
    pipe = get_llama_pipeline()

    prompt = f"""
    You are an HR AI system. Analyze the CV below and evaluate how well it matches the Job Description.
    Provide:
    - A score (0–100)
    - A short professional summary.

    Format strictly as:
    Score: <number>
    Summary: <short explanation>

    === Job Description ===
    {job_description}

    === CV ===
    {cv_text[:1800]}
    """

    output = pipe(prompt, max_new_tokens=250, temperature=0.3, do_sample=False)
    generated = output[0]["generated_text"]

    # Regex parsing
    score_match = re.search(r"Score\s*:\s*(\d{1,3})", generated)
    summary_match = re.search(r"Summary\s*:\s*(.*)", generated, re.DOTALL)

    if score_match:
        score = int(score_match.group(1))
    else:
        score = 0

    summary = summary_match.group(1).strip() if summary_match else "No summary found."
    score = min(max(score, 0), 100)

    return score, summary
