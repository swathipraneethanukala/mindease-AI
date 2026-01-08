import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = Flask(__name__)

# IMPORTANT: Strict CORS to allow your local file to talk to the server
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- Global Logic ---
retriever = None
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

def initialize_rag():
    global retriever
    # Ensure this path is 100% correct on your machine
    pdf_path = r"C:\Users\sama\Desktop\chatbot\data"
    
    if not os.path.exists(pdf_path):
        print(f"!!! CRITICAL ERROR: Path {pdf_path} not found. !!!")
        return

    print("--- Loading documents and initializing RAG ---")
    try:
        loader = PyPDFDirectoryLoader(pdf_path)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        
        print(f"Processing {len(docs)} document chunks...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings,
            persist_directory="./chroma_db_flask"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("--- RAG System Initialized and Ready ---")
    except Exception as e:
        print(f"Initialization Error: {e}")

# Health Check Route
@app.route("/", methods=["GET"])
def health_check():
    status = "Ready" if retriever else "Retriever Not Initialized"
    return f"Backend Online. Status: {status}", 200

# Main Chat Route
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat_endpoint():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if retriever is None:
        return jsonify({"error": "Retriever not initialized."}), 500

    data = request.json
    user_message = data.get("message")
    history_data = data.get("history", [])

    # Using the empathetic system prompt
    system_prompt = (
        "You are an empathetic and mood-aware Mental Health AI Companion designed to support youth. "
        "Acknowledge the user's mood gently, use the retrieved context for wellness strategies, and maintain a supportive tone.\n\n"
        "the response should be in markdown format without symbols and all but with headings and bullet points"
        "Always structure your response using clear headings\n"
        "Use bullet points (using - or •) for listing information\n"
        "Keep each bullet point concise and actionable\n"
        "CONTEXT FOR RETRIEVAL:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    chat_history = []
    for msg in history_data:
        role = "human" if msg.get("role") == "human" else "ai"
        chat_history.append((role, msg.get("content", "")))

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        response = rag_chain.invoke({
            "input": user_message,
            "chat_history": chat_history
        })
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print(f"Inference Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    initialize_rag()
    # Forces port 8000 and disables reloader to stop the "WinError 10038"
    app.run(host="127.0.0.1", port=8000, debug=True, use_reloader=False)