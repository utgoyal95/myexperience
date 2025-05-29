import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ClusterLabeler:
    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        umap_kwargs: dict = None,
        hdbscan_kwargs: dict = None,
        n_label_terms: int = 3
    ):
        self.embedder = SentenceTransformer(embed_model)
        self.reducer  = umap.UMAP(**(umap_kwargs or {
            "n_components": 5, "metric": "cosine", "random_state": 42
        }))
        self.clusterer = hdbscan.HDBSCAN(**(hdbscan_kwargs or {
            "min_cluster_size": 5,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        }))
        self.n_label_terms = n_label_terms
        self.cluster_names = {}

    def _discover_cluster_names(self, texts: list[str], labels: list[int]) -> dict[int, str]:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
        X = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()

        names = {}
        for cid in sorted(set(labels)):
            idx = [i for i, lbl in enumerate(labels) if lbl == cid]
            if not idx:
                names[cid] = f"cluster_{cid}"
                continue
            mean_tfidf = X[idx].mean(axis=0).A1
            top_n = np.argsort(mean_tfidf)[-self.n_label_terms:][::-1]
            top_terms = [feature_names[i] for i in top_n]
            names[cid] = "_".join(top_terms)
        return names

    def fit(self, file_paths: list[str]) -> pd.DataFrame:
        # 1) Load and concat using line-based reading
        dfs = []
        for path in file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            df = pd.DataFrame({'text': lines})
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)

        # 2) Embed
        texts = data["text"].tolist()
        embs  = self.embedder.encode(texts, show_progress_bar=True)

        # 3) Reduce & cluster
        reduced = self.reducer.fit_transform(embs)
        self.clusterer.fit(reduced)

        # 4) Assign numeric labels and scores
        data["cluster_id"] = self.clusterer.labels_
        data["score"] = self.clusterer.probabilities_

        # 5) Discover human-readable names
        self.cluster_names = self._discover_cluster_names(texts, data["cluster_id"].tolist())
        data["label"] = data["cluster_id"].map(lambda cid: self.cluster_names.get(cid, f"cluster_{cid}"))

        return data

    def predict(self, text: str) -> tuple[str, float]:
        emb   = self.embedder.encode([text])
        red   = self.reducer.transform(emb)
        cid   = self.clusterer.predict(red)[0]
        score = float(self.clusterer.probability(red)[0])
        name  = self.cluster_names.get(cid, f"cluster_{cid}")
        return name, score

    def save(self, path: str):
        # Save model objects and cluster_names mapping
        joblib.dump({
            'embedder': self.embedder,
            'reducer': self.reducer,
            'clusterer': self.clusterer,
            'cluster_names': self.cluster_names
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.embedder = data['embedder']
        self.reducer = data['reducer']
        self.clusterer = data['clusterer']
        self.cluster_names = data.get('cluster_names', {})