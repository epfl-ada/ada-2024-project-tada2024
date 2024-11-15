from sentence_transformers import SentenceTransformer


class All_MiniLM_L6_v2:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.model_info = "A sentence transformer optimized for generating embeddings of short paragraphs; It will use only the first 256 tokens of the input text."
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text):
        embedding = self.model.encode(text)
        return embedding


class all_mpnet_base_v2:
    def __init__(self):
        self.model_name = "all_mpnet_base_v2"
        self.model_info = ""
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def embed(self, text):
        embedding = self.model.encode(text)
        return embedding


class Roberta:
    def __init__(self):
        self.model_name = "roberta"
        self.model_info = "A sentence transformer optimized for generating embeddings of short paragraphs; It will use only the first 256 tokens of the input text."
        self.model = SentenceTransformer("stsb-roberta-large")

    def embed(self, text):
        embedding = self.model.encode(text)
        return embedding
