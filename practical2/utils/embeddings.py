from .vocab import Vocabulary
import numpy as np

EMBED_FILENAME_DICT = {
    "word2vec": "googlenews.word2vec.300d.txt",
    "glove": "glove.840B.300d.sst.txt",
}

def load_pt_embeddings(name="word2vec"):
    with open("data/"+EMBED_FILENAME_DICT[name], "r") as f:
        embeddings = f.readlines()

    # Set all token vectors to 0, <pad> can stay 0
    vectors = np.zeros((len(embeddings)+2, 300), dtype=np.float64)

    # Initialize <unk> token randomly
    vectors[0] = np.random.normal(0, 1/np.sqrt(300), size=300)

    v = Vocabulary()
    v.build()
    for i, line in enumerate(embeddings):    
        split = line.split(' ')
        token = split[0]
        vectors[i+2] = split[1:]
        v.add_token(token)
        
    print("Vocabulary size:", len(v.w2i))

    return v, vectors