# -----------------------------
# 1. Load dataset
# -----------------------------
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

ds = load_dataset("brunokreiner/genius-lyrics", split="train")

ds = ds[:128]

print(len(ds), "songs loaded")\


# We will embed ds["lyrics"]
lyrics = ds["lyrics"]
song_ids = ds["id"]

# -----------------------------
# 2. Load embedding model (GPU)
# -----------------------------
from sentence_transformers import SentenceTransformer

# Option A (Recommended): MPNet — fast, very good
model_name = "sentence-transformers/all-mpnet-base-v2"

model = SentenceTransformer(model_name)
model = model.to("cuda")

print("Model device:", next(model.parameters()).device)
torch.cuda.empty_cache()

# -----------------------------
# 3. Compute embeddings (GPU)
# -----------------------------
batch_size = 32

all_embs = []
for i in tqdm(range(0, len(lyrics), batch_size)):
    batch = lyrics[i : i + batch_size]

    # encode on GPU
    with torch.no_grad():
        emb = model.encode(batch,
                           device="cuda",
                           convert_to_numpy=True,
                           show_progress_bar=False)

    all_embs.append(emb)

embeddings = np.vstack(all_embs).astype("float32")
print("Embedding matrix shape:", embeddings.shape)

# -----------------------------
# 4. Normalize for cosine
# -----------------------------
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_norm = embeddings / norms

# -----------------------------
# 5. Save embeddings + metadata
# -----------------------------
import os
os.makedirs("emb_data_small", exist_ok=True)

np.save("emb_data_small/lyrics_embeddings.npy", emb_norm)

meta = pd.DataFrame({
    "song_id": song_ids,
    "artist": ds["artist_name"],
    "genre": ds["genres_list"],
    "popularity": ds["popularity"]
})
meta.to_parquet("emb_data_small/metadata.parquet", index=False)

print("Saved embeddings + metadata")