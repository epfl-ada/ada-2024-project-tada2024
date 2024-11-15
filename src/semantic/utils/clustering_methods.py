from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids


def Kmeans_Raw(data, n_clusters, state=520):
    kmeans = KMeans(n_clusters=n_clusters, random_state=state)
    kmeans.fit(data)
    return kmeans.labels_


def Kmedoids_Euc(data, n_clusters, state=520):
    kmedoids_euc = KMedoids(
        n_clusters=n_clusters, metric="euclidean", random_state=state
    )
    kmedoids_euc.fit(data)
    return kmedoids_euc.labels_


def Kmedoids_Man(data, n_clusters, state=520):
    kmedoids_man = KMedoids(
        n_clusters=n_clusters, metric="manhattan", random_state=state
    )
    kmedoids_man.fit(data)
    return kmedoids_man.labels_


def Kmedoids_Cos(data, n_clusters, state=520):
    kmedoids_cos = KMedoids(n_clusters=n_clusters, metric="cosine", random_state=state)
    kmedoids_cos.fit(data)
    return kmedoids_cos.labels_


def Spectral_NN(data, n_clusters, state=520):
    spectral_nn = SpectralClustering(
        n_clusters=n_clusters, affinity="nearest_neighbors", random_state=state
    )
    return spectral_nn.fit_predict(data)


def Spectral_RBF(data, n_clusters, state=520):
    spectral_rbf = SpectralClustering(
        n_clusters=n_clusters, affinity="rbf", random_state=state
    )
    return spectral_rbf.fit_predict(data)
