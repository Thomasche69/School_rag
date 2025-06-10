from flask import Flask, request, jsonify
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

app = Flask(__name__)

@app.route('/ollama', methods=['POST'])
def aiPost():
    """
    Endpoint to generate responses from the Ollama model.
    Expects a JSON payload with 'query' and 'model_name' keys.
    """
    print("Received request")
    json_content = request.json
    query = json_content.get('query', '')
    model_name = json_content.get('model_name', 'deepseek-r1:8b')  # Default model

    print(f"Query: {query}")
    print(f"Model: {model_name}")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
    # Dynamically load the specified model
       ollama_model = OllamaLLM(model=model_name)
    
    # Wrap the query in a list as required by the generate method
       llm_result = ollama_model.generate([query])  # Generate response
       response_answer = llm_result.generations[0][0].text  # Extract the generated text
       return jsonify({"response": response_answer})
    except Exception as e:
       print(f"Error: {e}")
       return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/embed', methods=['POST'])
def embed():
    """
    Endpoint to generate embeddings from the Ollama model.
    Expects a JSON payload with 'texts' and 'model_name' keys.
    """
    json_content = request.json
    texts = json_content.get('texts', [])
    model_name = json_content.get('model_name', 'deepseek-r1:8b')  # Default model

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    try:
        # Dynamically load the specified model
        embedding_model = OllamaEmbeddings(model=model_name)
        embeddings = embedding_model.embed_documents(texts)  # Generate embeddings for the texts
        return jsonify({"embeddings": embeddings})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

