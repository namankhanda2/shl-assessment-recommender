import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load catalog JSON
with open("./catalog.json", "r") as f:
    catalog = json.load(f)

# Convert catalog into searchable text
texts = []

for item in catalog:
    text = item["name"] + " " + item["description"]
    texts.append(text)

# Generate embeddings
embeddings = model.encode(texts)

# Convert embeddings to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(embeddings)

print("FAISS index created successfully!")
print("Total assessments indexed:", len(catalog))

# Example query
query = "Java backend developer"

# Convert query to embedding
query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")

# Search top 3 results
k = 3

distances, indices = index.search(query_embedding, k)

print("\nTop Recommendations:\n")

for i in indices[0]:
    print(catalog[i]["name"])