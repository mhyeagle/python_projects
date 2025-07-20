from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]
sentences_1 = ["千金方", "明月几时有"]

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(sentences_1)
print(embeddings)
