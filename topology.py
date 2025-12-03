import numpy as np
import pandas as pd

from tqdm import tqdm

# ------------------------------
# Load embeddings + metadata
# ------------------------------
embeddings = np.load("emb_data_small/lyrics_embeddings.npy")   # shape (N, d)
meta = pd.read_parquet("emb_data_small/metadata.parquet")

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

tda = ripser(
    dist_matrix,
    maxdim=1,
    distance_matrix=True,
    do_cocycles=True      # IMPORTANT: enables cocycle extraction
)

diagrams = tda["dgms"]
cocycles = tda["cocycles"][1]   # H1 cocycles
H1 = diagrams[1]
print("Found", len(H1), "H1 features")



# ============================================================
# (C) EXTRACT LOOP POINTS FROM COCYCLES
# ============================================================

def cocycle_to_points(cocycle, threshold=0.3):
    """
    Convert a cocycle (list of (i,j,w) tuples) into a set of points
    likely part of the persistent cycle.
    Edges with |w| > threshold are considered significant.
    """
    pts = set()
    for (i, j, w) in cocycle:
        if abs(w) > threshold:
            pts.add(i)
            pts.add(j)
    return list(pts)


# ============================================================
# (D) GET TOP PERSISTENT H1 LOOPS
# ============================================================

pers = H1[:,1] - H1[:,0]
K = min(5, len(H1))         # Show top 5 loops (or fewer if dataset small)
top_idx = np.argsort(pers)[-K:][::-1]

print("\nTop H1 features sorted by persistence:")
for idx in top_idx:
    print(f"  Feature {idx}: persistence = {pers[idx]:.4f}")


# ============================================================
# (E) STORE METADATA SUMMARY FOR EACH LOOP IN A CSV
# ============================================================

# Initialize a list to store results
results = []

for rank, hidx in enumerate(top_idx):
    birth, death = H1[hidx]
    cocycle = cocycles[hidx]
    pts = cocycle_to_points(cocycle, threshold=0.3)

    # Skip if no significant edges are found
    if len(pts) == 0:
        continue

    loop_meta = meta.iloc[pts]

    # Collect data for this loop
    result = {
        "loop_rank": rank + 1,
        "h1_feature_index": hidx,
        "birth": birth,
        "death": death,
        "persistence": pers[hidx],
        "num_points_in_loop": len(pts),
        "top_genres": ", ".join(loop_meta["genre"].value_counts().head().index),
        "top_artists": ", ".join(loop_meta["artist"].value_counts().head().index),
        "sample_songs": "; ".join(
            loop_meta[["artist", "lyrics"]]
            .apply(lambda row: f"{row['artist']} — {row['lyrics'][:100]}...", axis=1)
            .head()
        ),
    }

    results.append(result)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("h1_loops_summary.csv", index=False, encoding="utf-8")

print("Results saved to h1_loops_summary.csv")

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