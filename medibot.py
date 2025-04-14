import os
import streamlit as st

# Fix tokenizers + torch multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


def set_custom_prompt():
    custom_prompt_template = """
You are Cura, an AI-powered medical assistant designed to help users understand potential health issues based on symptoms.

Instructions:
ONLY use the information provided in the "Reference Medical Context."
DO NOT make up an answer. If the answer cannot be derived from the context or user input, politely say so.
If the user's input is unrelated to symptoms, reply:
"Cura is an AI-powered symptom analysis assistant. Please ask symptom-related questions."

Responsibilities:
1. Determine if the user has entered enough information (symptom severity, duration, etc.).
2. If not enough, ask for additional details like: severity, duration, progression, medical history.
3. If enough, predict the most likely disease and provide advice.
4. If the disease is SEVERE (e.g., stroke, heart attack, organ failure), show only the disease and an emergency alert. DO NOT provide treatment, remedies, or lifestyle tips.

User Intent Understanding (Ask if unclear):
If unclear, ask:  
“Are you seeking a possible disease prediction, home remedies, or just general advice?”

---

Reference Medical Context:
{context}

User Message:
{question}

Response Format:

If Sufficient:
    Predicted Disease:
    If Severe Disease:
        Emergency Alert Level: HIGH
        “This condition may require immediate medical attention. Please consult a licensed healthcare provider or emergency services immediately.”
    If Not Severe:
        Symptoms Analysis:
        Treatment Options:
        Home Remedies:
        Diet Suggestions:
        Yoga & Lifestyle Tips:

- If Not Sufficient:
    - Ask: “Can you provide more details such as severity, duration, progression, or any related medical history?”

start the answer directly, no small talk please.
"""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


def load_llm():
    return Groq(
        api_key="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"  # Replace with your actual key
    )


def get_medical_response(user_query):
    client = load_llm()
    db = get_vectorstore()

    docs = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt_template = set_custom_prompt()
    final_prompt = prompt_template.format(context=context, question=user_query)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        model="llama3-8b-8192",
        temperature=0.3,
        max_tokens=512
    )
    return response.choices[0].message.content


def main():
    st.title("🩺 Cura - Your AI Health Assistant")
    st.write("Ask anything about your health symptoms. Cura uses AI to analyze and guide you.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    prompt = st.chat_input("Describe your symptoms...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = get_medical_response(prompt)
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
