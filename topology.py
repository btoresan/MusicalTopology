import numpy as np
import pandas as pd

# ------------------------------
# Load embeddings + metadata
# ------------------------------
embeddings = np.load("emb_data/lyrics_embeddings.npy")   # shape (N, d)
meta = pd.read_parquet("emb_data/metadata.parquet")

print("Embeddings:", embeddings.shape)
print(meta.head())

###############################################################################
# 1. CPU KNN (sklearn)
###############################################################################
from sklearn.neighbors import NearestNeighbors

k = 30
knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
knn.fit(embeddings)
distances, indices = knn.kneighbors(embeddings)

###############################################################################
# 2. Build distance matrix from KNN graph
###############################################################################

N = embeddings.shape[0]
dist_matrix = np.full((N, N), np.inf, dtype=np.float32)

for i in range(N):
    for j, d in zip(indices[i], distances[i]):
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

np.fill_diagonal(dist_matrix, 0.0)

###############################################################################
# 3. Run topological analysis (Ripser)
###############################################################################
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

print("Running Ripser...")
tda = ripser(dist_matrix, maxdim=1, distance_matrix=True)
diagrams = tda["dgms"]

plt.figure(figsize=(8,4))
plot_diagrams(diagrams)
plt.show()

###############################################################################
# 4. Persistence Entropy
###############################################################################
import numpy as np

def persistent_entropy(diagram):
    # Remove points with infinite death (optional: keep or ignore depending on your application)
    finite = diagram[np.isfinite(diagram[:,1])]
    if len(finite) == 0:
        return 0.0

    lengths = finite[:,1] - finite[:,0]
    L = np.sum(lengths)
    if L == 0:
        return 0.0

    p = lengths / L  # probabilities
    entropy = -np.sum(p * np.log(p))
    return entropy

print("Persistent Entropies:")
for dim, diag in enumerate(diagrams):
    H = persistent_entropy(diag)
    print(f"H{dim} entropy:", H)


###############################################################################
# 5. Per-genre topology
###############################################################################

from sklearn.metrics import pairwise_distances

genres = meta["genre"].fillna("null")

print("\nTDA per genre:")
for g in genres.unique():
    idx = np.where(genres == g)[0]

    if len(idx) < 10:
        continue  # need enough samples

    emb_g = embeddings[idx]
    dist_g = pairwise_distances(emb_g)

    dgm_g = ripser(dist_g, maxdim=1, distance_matrix=True)["dgms"]
    h1 = dgm_g[1]  # 1-dimensional holes

    print(f"{g:20s} | samples: {len(idx):4d} | H1 loops: {len(h1)}")

print("\nDone.")
