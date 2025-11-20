#import minepy
import pandas as pd
import numpy as np
import minepy

def top_k(X, y, n):
    m = minepy.MINE()
    mic_scores = {}
    for col in X.columns:
        m.compute_score(X[col], y)
        mic_scores[col] = m.mic()
    selected_features = sorted(mic_scores, key=mic_scores.get, reverse=True)
    top_k = selected_features[:n]
    return top_k