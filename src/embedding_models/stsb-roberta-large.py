from sentence_transformers import SentenceTransformer

class All_MiniLM_L6_v2:
    def __init__(self):
        self.model_name = 'stsb-roberta-large'
        self.model_info = "A sentence transformer optimized for generating embeddings of short paragraphs; It will use only the first 256 tokens of the input text."
        self.model =  SentenceTransformer('stsb-roberta-large')


    def embed(self, text):
        embedding = self.model.encode(text)
        return embedding
