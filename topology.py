import numpy as np
import pandas as pd

embeddings = np.load("emb_data/lyrics_embeddings.npy")   # (N, d)
meta = pd.read_parquet("emb_data/metadata.parquet")

print("Embeddings:", embeddings.shape)
meta.head()
