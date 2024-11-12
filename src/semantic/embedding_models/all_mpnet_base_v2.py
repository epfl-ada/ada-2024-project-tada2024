from sentence_transformers import SentenceTransformer

class all_mpnet_base_v2:
    def __init__(self):
        self.model_name = 'all_mpnet_base_v2'
        self.model_info = ""
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
   
    def embed(self, text):
        embedding = self.model.encode(text)
        return embedding