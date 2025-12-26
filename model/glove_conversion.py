import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

# Source: https://nlp.stanford.edu/projects/glove/

txt = Path(r"C:\tmp\wiki_giga_2024.txt")
h5  = r"C:\tmp\latent50.h5"
meta = Path(r"C:\tmp\meta.csv")

cols = ['word'] + [f'c{i}' for i in range(50)]

good_rows = []
bad = []  # (lineno, count, sample)

with txt.open('r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f, 1):
        parts = line.strip().split()
        if len(parts) == 51:
            good_rows.append([parts[0]] + [float(x) for x in parts[1:]])
        else:
            bad.append((i, len(parts), line[:120]))

if bad:
    print(f"{len(bad)} malformed lines, e.g.: {bad[:5]}")

df = pd.DataFrame(good_rows, columns=cols)

# 2) Ensure numeric vectors
vecs = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
good = vecs.notnull().all(axis=1)
bad_rows = (~good).sum()
if bad_rows:
    print(f"Dropping {bad_rows} malformed rows.")
df = pd.concat([df[['word']], vecs], axis=1)[good]

df = df.sample(n=5000, random_state=42)

print(df.head())

# 3) Save to HDF5 (table format is nice for later filtering)
df.to_hdf(h5, key="latent", format="table", mode="w")


# ---- later / elsewhere ----
df = pd.read_hdf(h5, key="latent")

X = np.copy(df)

# 4) Build a numeric matrix for sklearn
X = df.filter(regex=r"^c\d+$").to_numpy(dtype=np.float32)  # shape: (n_words, 50)

# 5) Pairwise cosine + t-SNE (warning: O(n^2) memory/time)
dists = pairwise_distances(X, metric="cosine")
tsne = TSNE(n_components=2, verbose=1, perplexity=30)
emb = tsne.fit_transform(dists)

# Optional: keep the words alongside the 2D embedding
tsne_df = pd.DataFrame({"word": df["word"], "x": emb[:,0], "y": emb[:,1]})
print(tsne_df.head())

# Save metadata .csv file
# df is your DataFrame shown above
# meta is your target path (e.g., "meta" directory)

df_meta = df.reset_index(drop=True).rename(columns={"word": "name"})[["name"]]
df_meta.insert(0, "i", range(len(df_meta)))
df_meta.to_csv(meta, index=False)
