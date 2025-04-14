from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Fix tokenizers + torch multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "https://cura-health-compass-main-3-h3pkd7d7g3l.vercel.app",
            "https://cura-health-compass-main-3-bzpwhymo6.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def get_vectorstore():
    try:
        print("Loading vectorstore from disk...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(current_dir, "vectorstore", "db_faiss")
        print(f"Looking for vectorstore at: {index_path}")
        
        if not os.path.exists(index_path):
            print("Vectorstore not found, creating new one...")
            return create_new_vectorstore()
            
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("Vectorstore loaded successfully")
        return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {str(e)}")
        raise

def create_new_vectorstore():
    try:
        print("Creating new vectorstore...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(current_dir, "data", "encyclopedia-of-medicine-vol-1-5-3rd-edition copy.pdf")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)
        
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Save the vectorstore
        index_path = os.path.join(current_dir, "vectorstore", "db_faiss")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vectorstore.save_local(index_path)
        
        print("New vectorstore created and saved successfully")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {str(e)}")
        raise

def set_custom_prompt():
    custom_prompt_template = """
You are Cura, an AI-powered medical assistant. Analyze the symptoms and respond in one of two ways:

1. If symptoms information is INSUFFICIENT:
Ask specific questions about missing details, formatted as bullet points:
- "What is the severity of your [symptom]? (mild, moderate, severe)"
- "How long have you been experiencing these symptoms?"
- "Are the symptoms constant or do they come and go?"
- "Are the symptoms getting better or worse?"
- "Do you have any relevant medical history or conditions?"

2. If symptoms information is SUFFICIENT:
Provide a structured response in exactly this format:

Based on the provided information, I will analyze the symptoms.

Predicted Disease: [Name of the most likely condition]

Symptoms Analysis: [Brief analysis of the symptoms and why they suggest this diagnosis]

Treatment Options: [Possible medical treatments, medications, or interventions]

Home Remedies: [Safe self-care measures and home treatments]

Diet Suggestions: [Specific foods to eat or avoid]

Yoga & Lifestyle Tips: [Exercise, rest, or lifestyle modifications]

Please note that this is a predicted diagnosis based on the provided information. It is essential to consult a licensed healthcare provider for an accurate diagnosis and personalized treatment plan.

3. If symptoms suggest a SEVERE condition:
Only respond with:
Emergency Alert Level: HIGH
[Condition name]
This condition requires immediate medical attention. Please seek emergency care immediately.

Reference Medical Context:
{context}

User Message:
{question}

Respond directly without any introductory phrases.
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

# Cache for vectorstore and prompt template
_vectorstore = None
_prompt_template = None

def get_medical_response(user_query):
    global _vectorstore, _prompt_template
    
    try:
        # Initialize cached objects if not already done
        if _vectorstore is None:
            print("Initializing vectorstore...")
            _vectorstore = get_vectorstore()
        if _prompt_template is None:
            print("Initializing prompt template...")
            _prompt_template = set_custom_prompt()

        # Load LLM client
        print("Loading LLM client...")
        client = load_llm()

        # Get relevant context with reduced k for faster search
        print("Searching for relevant context...")
        docs = _vectorstore.similarity_search(user_query, k=2)
        context = "\n".join([doc.page_content for doc in docs])

        # Format prompt
        print("Formatting prompt...")
        final_prompt = _prompt_template.format(context=context, question=user_query)

        # Get response with optimized parameters
        print("Getting response from LLM...")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model="llama3-8b-8192",
            temperature=0.2,  # Reduced for more focused responses
            max_tokens=384,   # Reduced for faster responses
            top_p=0.9        # Added to improve response quality while maintaining speed
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in get_medical_response: {str(e)}")
        raise

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        message = data['message']
        
        # Special case for ping
        if message.lower() == 'ping':
            return jsonify({'response': 'pong'}), 200
            
        print(f"Received message: {message}")
        
        try:
            response = get_medical_response(message)
            print(f"Generated response: {response}")
            return jsonify({'response': response}), 200
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
