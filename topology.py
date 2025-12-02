import numpy as np
import pandas as pd

from tqdm import tqdm

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

from scipy.sparse import csr_matrix
import numpy as np

rows = []
cols = []
vals = []

N = embeddings.shape[0]
print("Calculating dist matrix")
for i in tqdm(range(N)):
    for j, d in zip(indices[i], distances[i]):
        rows.append(i)
        cols.append(j)
        vals.append(d)
        rows.append(j)
        cols.append(i)
        vals.append(d)

# Build sparse symmetric matrix
dist_matrix = csr_matrix((vals, (rows, cols)), shape=(N, N))


###############################################################################
# 3. Run topological analysis (Ripser)
###############################################################################
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

print("Running Ripser...")
tda = ripser(dist_matrix, maxdim=1, distance_matrix=True)

diagrams = tda["dgms"]
pairs = {dim: diag for dim, diag in enumerate(diagrams)}


#plt.figure(figsize=(8,4))
#plot_diagrams(diagrams)
#plt.show()

output = []

output.append("===== TOPOLOGICAL ANALYSIS REPORT =====\n")

# -----------------------------------------
# H1 (loops)
# -----------------------------------------
H1 = diagrams[1]
H1_pairs = pairs[1]

output.append("\n=== H1 (1-dimensional holes / loops) ===\n")

for (b, d), (birth_simplex, death_simplex) in zip(H1, H1_pairs):
    pers = d - b
    if pers < 0.05:  # threshold for meaningful loops – adjust if needed
        continue
    
    output.append(f"\nH1 feature – persistence {pers:.4f}")
    
    # birth simplex
    output.append("  Birth simplex points:")
    for idx in birth_simplex:
        row = meta.iloc[idx]
        output.append(f"    - {row['artist']} — {row['song_id']} — genre={row['genre']}")
    
    # death simplex
    output.append("  Death simplex points:")
    for idx in death_simplex:
        row = meta.iloc[idx]
        output.append(f"    - {row['artist']} — {row['song_id']} — genre={row['genre']}")

# -----------------------------------------
# H0 (connected components)
# -----------------------------------------
H0 = diagrams[0]
H0_pairs = pairs[0]

output.append("\n=== H0 (connected components) ===\n")

for (b, d), (edge, _) in zip(H0, H0_pairs):
    pers = d - b
    if pers < 0.05:
        continue
    
    output.append(f"\nH0 feature – persistence {pers:.4f}")
    
    i, j = edge
    for idx in [i, j]:
        row = meta.iloc[idx]
        output.append(f"    - {row['artist']} — {row['song_id']} — genre={row['genre']}")

# -----------------------------------------
# Save to TXT
# -----------------------------------------
with open("tda_features.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Saved meaningful persistent features to tda_features.txt")

###############################################################################
# 4. Persistence Entropy
###############################################################################
import numpy as np

def persistent_entropy_stream(diagram):
    total_length = 0.0

    # First pass: compute total persistence (L)
    for birth, death in diagram:
        if np.isfinite(death):
            total_length += (death - birth)

    if total_length == 0:
        return 0.0

    # Second pass: compute entropy
    entropy = 0.0
    for birth, death in diagram:
        if np.isfinite(death):
            l = death - birth
            p = l / total_length
            entropy -= p * np.log(p)

    return entropy

print("Persistent Entropies:")
for dim, diag in enumerate(diagrams):
    H = persistent_entropy_stream(diag)
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
        continue  # minimum samples for meaningful TDA

    if len(idx) > 3000:
        idx = np.random.choice(idx, 3000, replace=False)


    emb_g = embeddings[idx]

    # Use Ripser on point cloud directly (no N×N matrix!)
    tda_g = ripser(emb_g, maxdim=1)
    dgm_g = tda_g["dgms"]
    
    h1 = dgm_g[1]

    print(f"{g:20s} | samples: {len(idx):4d} | H1 loops: {len(h1)}")

print("\nDone.")

