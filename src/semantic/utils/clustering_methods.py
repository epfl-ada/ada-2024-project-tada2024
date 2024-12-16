from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids


def Kmedoids_Man(data, n_clusters, state=520):
    kmedoids_man = KMedoids(
        n_clusters=n_clusters, metric="manhattan", random_state=state, init="k-medoids++", 
    )
    kmedoids_man.fit(data)
    return kmedoids_man.labels_


def Kmedoids_Cos(data, n_clusters, state=520):
    kmedoids_cos = KMedoids(n_clusters=n_clusters, metric="cosine", random_state=state,init="k-medoids++")
    kmedoids_cos.fit(data)
    return kmedoids_cos.labels_