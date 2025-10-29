import os
import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig

@st.cache_resource
def get_llama_pipeline():
    """Load base LLaMA 3.1 model + PEFT adapter for CV-job matching."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("❌ Missing HUGGINGFACEHUB_API_TOKEN in environment!")

    base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    peft_model_id = "LlamaFactoryAI/Llama-3.1-8B-Instruct-cv-job-description-matching"

    print("🔹 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=token,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)

    print("🔹 Loading PEFT adapter...")
    config = PeftConfig.from_pretrained(peft_model_id, token=token)
    model = PeftModel.from_pretrained(base_model, peft_model_id, token=token)

    print("✅ Model loaded successfully!")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def score_with_llama(cv_text, job_description):
    """Use the fine-tuned LLaMA PEFT model to score CV-job description fit."""
    pipe = get_llama_pipeline()

    prompt = f"""
    You are an AI HR system fine-tuned to evaluate CV-job matching.
    Compare the CV below with the provided job description.

    Return the result in the following format:
    Score: <number from 0–100>
    Summary: <1–3 sentences summarizing the reasoning>

    === Job Description ===
    {job_description}

    === Candidate CV ===
    {cv_text[:1800]}
    """

    output = pipe(
        prompt,
        max_new_tokens=250,
        temperature=0.2,
        do_sample=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    generated = output[0]["generated_text"]

    # Regex-based safe parsing
    score_match = re.search(r"Score\s*:\s*(\d{1,3})", generated)
    summary_match = re.search(r"Summary\s*:\s*(.*)", generated, re.DOTALL)

    score = int(score_match.group(1)) if score_match else 0
    summary = summary_match.group(1).strip() if summary_match else "No summary found."
    score = min(max(score, 0), 100)

    return score, summary
