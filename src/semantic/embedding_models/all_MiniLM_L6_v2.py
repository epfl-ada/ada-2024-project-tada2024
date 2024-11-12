from sentence_transformers import SentenceTransformer

class All_MiniLM_L6_v2:
    def __init__(self):
        self.model_name = 'all-MiniLM-L6-v2'
        self.model_info = "A sentence transformer optimized for generating embeddings of short paragraphs; It will use only the first 256 tokens of the input text."
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def embed(self, text):
        embedding = self.model.encode(text)
        return embedding