import streamlit as st
import fitz
from openai import OpenAI
import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

MODELS = {
    "Meta Llama 3/70b (Free version)": {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "cost_per_token": 0.00,
    },
    "Claude 3.5 Sonnet": {
        "id": "anthropic/claude-3.5-sonnet",
        "cost_per_token": 0.00000727,
    },
    "GPT-4o mini": {"id": "openai/gpt-4o-mini", "cost_per_token": 0.000000328},
}

st.title("AI Resume Parser & Benchmark Tool")

model_choice = st.selectbox("Choose an AI Model", list(MODELS.keys()))
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt", "docx"])

PROMPT_FILE = "prompt.txt"
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        default_prompt = f.read()
else:
    default_prompt = "You are a resume parsing and analysis assistant."

user_prompt = st.text_area("AI Prompt (editable)", value=default_prompt, height=200)
additional_instructions = st.text_area(
    "Additional instructions for the AI (optional)", ""
)

LOG_FILE = "resume_ai_log.csv"


def save_log(entry):
    new_entry_df = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
        df_log = pd.concat([df_log, new_entry_df], ignore_index=True)
    else:
        df_log = new_entry_df
    df_log.to_csv(LOG_FILE, index=False)


def calculate_cost(model_choice, input_tokens, output_tokens):
    cost_per_token = MODELS[model_choice]["cost_per_token"]
    return (input_tokens + output_tokens) * cost_per_token


if st.button("Analyze Resume"):
    if uploaded_file:
        text_content = ""
        if uploaded_file.name.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                text_content += page.get_text()
        else:
            text_content = uploaded_file.read().decode("utf-8", errors="ignore")

        full_prompt = f"{user_prompt}\n\nResume:\n{text_content}\n\nAdditional instructions:\n{additional_instructions}"

        with st.spinner("Sending request to AI model... This may take a few seconds."):
            start_time = time.time()
            completion = client.chat.completions.create(
                model=MODELS[model_choice]["id"],
                messages=[{"role": "user", "content": full_prompt}],
            )
            end_time = time.time()
            elapsed_time = end_time - start_time

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

        cost = calculate_cost(model_choice, prompt_tokens, completion_tokens)

        response_text = completion.choices[0].message.content
        st.subheader("AI Response")
        st.code(response_text)

        st.subheader("Token Usage")
        st.write(f"Prompt tokens: {prompt_tokens}")
        st.write(f"Completion tokens: {completion_tokens}")
        st.write(f"Total tokens: {total_tokens}")

        st.subheader("Response Time")
        st.write(f"{elapsed_time:.2f} seconds")

        st.subheader("Estimated Cost")
        st.write(f"${cost:.4f} USD for this request")

        log_entry = {
            "model": model_choice,
            "resume_file": uploaded_file.name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "response_time_sec": round(elapsed_time, 2),
            "estimated_cost_usd": round(cost, 4),
            "response_snippet": response_text[:200].replace("\n", " "),
        }
        save_log(log_entry)
        st.success("✅ Analysis complete and logged!")

if os.path.exists(LOG_FILE):
    st.subheader("Benchmark Log")
    df_log = pd.read_csv(LOG_FILE)
    st.dataframe(df_log)

    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="resume_ai_log.csv",
        mime="text/csv",
    )
