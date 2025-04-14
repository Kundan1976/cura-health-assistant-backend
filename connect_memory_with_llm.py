import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize Groq client
def load_llm():
    return Groq(
        api_key="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"  # Replace with your actual API key
    )

# Custom prompt template
custom_prompt_template = """
You are Cura, an AI-powered medical assistant designed to help users understand potential health issues based on symptoms.

Instructions:
- ONLY use the information provided in the "Reference Medical Context."
- DO NOT make up an answer. If the answer cannot be derived from the context or user input, politely say so.
- If the user's input is unrelated to symptoms, reply:
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

## Reference Medical Context:
{context}

## User Message:
{question}

## Response Format:

- If Sufficient:
    - Predicted Disease:
    - If Severe Disease:
        - Emergency Alert Level: HIGH
        - “This condition may require immediate medical attention. Please consult a licensed healthcare provider or emergency services immediately.”
    - If Not Severe:
        - Symptoms Analysis:
        - Treatment Options:
        - Home Remedies:
        - Diet Suggestions:
        - Yoga & Lifestyle Tips:

- If Not Sufficient:
    - Ask: “Can you provide more details such as severity, duration, progression, or any related medical history?”

start the answer directly, no small talk please.
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# Load FAISS vectorstore
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vectorstore/db_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    )

# Main function
def get_medical_response(user_query):
    client = load_llm()
    db = load_vectorstore()

    # Retrieve relevant context
    docs = db.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Format the prompt
    prompt = set_custom_prompt().format(
        context=context,
        question=user_query
    )

    # Get response from Groq LLM
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        temperature=0.3,
        max_tokens=512
    )
    return response.choices[0].message.content

# CLI Chat Loop
if __name__ == "__main__":
    print("\n======== CURA CHAT MODE ========")
    print("🩺 Welcome to Cura — your AI health assistant!")
    print("Type 'exit' anytime to end the conversation.\n")

    message_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Cura: Take care! Wishing you good health. 💙")
            break

        try:
            message_history.append({"role": "user", "content": user_input})
            response = get_medical_response(user_input)
            print("\nCura:", response, "\n")
            message_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Cura: I'm sorry, I couldn't process that. Please try again.")
            print("source doucument:", response["source_documents"])