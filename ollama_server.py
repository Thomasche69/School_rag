from flask import Flask, request, jsonify
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

app = Flask(__name__)

MODEL_CACHE = {}
EMBEDDING_CACHE = {}
DEFAULT_MODEL = "qwen3:8b"

def get_llm(model_name):
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = ChatOllama(model=model_name)
    return MODEL_CACHE[model_name]

def get_embedding():
    if "nomic-embed-text:latest" not in EMBEDDING_CACHE:
        EMBEDDING_CACHE["nomic-embed-text:latest"] = OllamaEmbeddings(model="nomic-embed-text:latest")
    return EMBEDDING_CACHE["nomic-embed-text:latest"]

@app.route('/ollama', methods=['POST'])
def aiPost():
    print("Received request")
    json_content = request.json
    messages = json_content.get('messages', [])  # Receive messages array
    model_name = json_content.get('model_name', DEFAULT_MODEL)

    print(f"Messages: {messages}")
    print(f"Model: {model_name}")

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    try:
        ollama_model = get_llm(model_name)
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "human")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "ai":
                langchain_messages.append(AIMessage(content=content))
            else:  # Default to human for user messages
                langchain_messages.append(HumanMessage(content=content))
        
        # Use invoke for chat models
        result = ollama_model.invoke(langchain_messages)
        response_answer = result.content
        
        return jsonify({"response": response_answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/embed', methods=['POST'])
def embed():
    json_content = request.json
    texts = json_content.get('texts', [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    try:
        embedding_model = get_embedding()
        embeddings = embedding_model.embed_documents(texts)
        return jsonify({"embeddings": embeddings})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8082)
