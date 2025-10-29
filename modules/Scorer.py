import json
import re
from transformers import pipeline

def score_with_llama(cv_text, job_description):
    # Load model
    model = pipeline("text-generation", model="meta-llama/Llama-3-8b", device_map="auto")

    # Prompt ringkas, tidak perlu mention JSON literal
    prompt = f"""
    You are an AI HR recruiter. Analyze the following CV compared to the given Job Description.
    Give:
    1. a numerical score (0–100) representing match quality.
    2. a short, professional summary of the reasoning.

    Format your response strictly like:
    Score: <number>
    Summary: <text>

    === Job Description ===
    {job_description}

    === Candidate CV ===
    {cv_text[:2000]}
    """

    # Generate
    output = model(prompt, max_new_tokens=256, temperature=0.2)
    raw_text = output[0]["generated_text"]

    # Parse dengan regex
    score_match = re.search(r"Score\s*:\s*(\d{1,3})", raw_text)
    summary_match = re.search(r"Summary\s*:\s*(.*)", raw_text, re.DOTALL)

    if score_match:
        score = int(score_match.group(1))
    else:
        score = 0

    summary = summary_match.group(1).strip() if summary_match else "No summary found."

    # Clamp score biar aman
    score = min(max(score, 0), 100)

    return score, summary
