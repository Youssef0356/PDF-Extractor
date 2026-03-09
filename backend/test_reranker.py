import ollama

def test_reranker():
    query = "What is the operating pressure?"
    doc = "The device operates at a pressure of 5 bar and a temperature of 20°C."
    
    # Try chat API with a formatted prompt (many rerankers in Ollama use this)
    try:
        print("Testing Chat API:")
        response = ollama.chat(
            model='dengcao/Qwen3-Reranker-0.6B:Q8_0', 
            messages=[{'role': 'user', 'content': f'Query: {query}\nDocument: {doc}'}]
        )
        print(response)
    except Exception as e:
        print("Chat API Error:", e)

if __name__ == "__main__":
    test_reranker()
