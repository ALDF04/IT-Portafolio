"""Similarity search utilities for historical setups."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class SetupSimilarity:
    def __init__(self, features: pd.DataFrame):
        self.features = features
        self.scaler = StandardScaler()
        self.normalized = self.scaler.fit_transform(features)

    def find_similar(self, sample: pd.Series, top_k: int = 3) -> list[tuple[int, float]]:
        sample_norm = self.scaler.transform(sample.to_frame().T)
        sims = cosine_similarity(sample_norm, self.normalized)[0]
        idx_sorted = np.argsort(sims)[::-1][:top_k]
        return [(int(idx), float(sims[idx])) for idx in idx_sorted]
