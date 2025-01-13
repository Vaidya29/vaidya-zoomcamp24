import google.generativeai as genai
import google.cloud.aiplatform as aip
import requests

# Set your API key
genai.configure(api_key="AIzaSyCOpSLT4ie9pzRdSSTBwdq8o1o8UfPIm4M")

# Function to generate embeddings using the text-embedding-004 model
def generate_embeddings(text):
    response = genai.embed_content(model="models/text-embedding-004", content=text)
    return response["embedding"]
def generate_text(prompt):
    headers = {
        "Authorization": f"Bearer {"AIzaSyCOpSLT4ie9pzRdSSTBwdq8o1o8UfPIm4M"}"
    }
    data = {
        "prompt": prompt,
        "model": "models/gemini-pro"
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    try:
        return response.json()["choices"][0]["message"]["content"]  # Try to access choices first
    except KeyError:
        # If 'choices' is not present, return the full response text
        return response.json().get("text", "")  # Use get() with default value for 'text'

# Function to search using Google Embeddings and return top N results
def search_embeddings(query_embedding, embeddings_list, top_n=5):
    # Implement your similarity search algorithm here
    # You can use libraries like FAISS or ScaNN for efficient similarity search.
    # For simplicity, we'll use a naive cosine similarity approach:
    import numpy as np

    query_embedding = np.array(query_embedding)
    embeddings_list = np.array(embeddings_list)

    similarities = np.dot(embeddings_list, query_embedding) / (np.linalg.norm(embeddings_list, axis=1) * np.linalg.norm(query_embedding))
    top_indices = np.argsort(similarities)[::-1][:top_n]

    return top_indices

# Example usage
# Assuming you have a list of documents and their embeddings:
documents = [
    "Document 1: This is the first document.",
    "Document 2: This is the second document.",
    # ...
]

embeddings = [generate_embeddings(doc) for doc in documents]

# Get the query embedding
query = "What is the meaning of life?"
query_embedding = generate_embeddings(query)

# Search for similar documents
top_results = search_embeddings(query_embedding, embeddings)

# Retrieve the top N documents
for index in top_results:
    print(documents[index])

# Call the Gemini LLM to process the retrieved documents
prompt = "Given the following documents, answer the query: " + query + "\n\n" + "\n\n".join(documents[index] for index in top_results)
response = generate_text(prompt=prompt)
print(response.text)