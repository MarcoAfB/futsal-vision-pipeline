import numpy as np


def reduceFeatures(features, clean, config):
    if config["reduction"] == "none":
        return np.asarray(features, dtype=float)
    from umap import UMAP

    reducer = UMAP(n_components=config["umapComponents"], n_neighbors=min(config["umapNeighbors"], int(clean.sum()) - 1),
                   min_dist=config["umapMinDistance"], metric="cosine", random_state=config["seed"])
    reducer.fit(features[clean])
    return np.asarray(reducer.transform(features), dtype=float)


def clusterFeatures(features, clean, config):
    """Fit one match on its clean crops and assign every crop of that match.

    Returns cluster labels, cluster centres and the reduced points in which both live.
    """
    from sklearn.cluster import KMeans

    if int(clean.sum()) < max(config["minCleanCrops"], config["clusterCount"]):
        raise ValueError("Too few clean athlete crops to cluster a match.")
    points = reduceFeatures(features, clean, config)
    model = KMeans(n_clusters=config["clusterCount"], n_init=10, random_state=config["seed"]).fit(points[clean])
    return model.predict(points), model.cluster_centers_, points
