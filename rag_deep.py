from flask import Flask, request, jsonify
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

app = Flask(__name__)

MODEL_CACHE = {}
EMBEDDING_CACHE = {}
DEFAULT_MODEL = "qwen3:8b"

def get_llm(model_name):
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = OllamaLLM(model=model_name)
    return MODEL_CACHE[model_name]

def get_embedding():
    if "qwen3-embedding:0.6b" not in EMBEDDING_CACHE:
        EMBEDDING_CACHE["qwen3-embedding:0.6b"] = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    return EMBEDDING_CACHE["qwen3-embedding:0.6b"]

@app.route('/ollama', methods=['POST'])
def aiPost():
    print("Received request")
    json_content = request.json
    query = json_content.get('query', '')
    model_name = json_content.get('model_name', DEFAULT_MODEL)

    print(f"Query: {query}")
    print(f"Model: {model_name}")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        ollama_model = get_llm(model_name)
        llm_result = ollama_model.generate([query])
        response_answer = llm_result.generations[0][0].text
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
    serve(app, host="0.0.0.0", port=8080)
