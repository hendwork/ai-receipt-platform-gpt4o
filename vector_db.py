import numpy as np

def calculate_cosine_similarity(vec1, vec2):
    """Implementasi rumus cosine similarity."""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    dot_product = np.dot(vec1, vec2)
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    return dot_product / (norm_vec1 * norm_vec2)

class SimpleVectorDB:
    """Database sederhana untuk menyimpan vektor item."""
    def __init__(self):
        self.vectors = {}

    def add_item(self, item_id, vector, metadata=None):
        self.vectors[item_id] = {
            "vector": np.asarray(vector),
            "metadata": metadata if metadata is not None else {}
        }

    def find_similar_items(self, query_vector, k=3):
        results = []
        for item_id, data in self.vectors.items():
            similarity = calculate_cosine_similarity(query_vector, data["vector"])
            results.append((item_id, similarity, data["metadata"]))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]