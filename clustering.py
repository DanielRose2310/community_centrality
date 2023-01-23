import networkx as nx

def clustering_algo(_G,_clustering,_weight):
    comms = None
    if _clustering=='louvain':
        comms = nx.algorithms.community.louvain.louvain_communities(_G,weight=_weight)
    elif _clustering=='gn':
        comms_generator = nx.algorithms.community.girvan_newman(_G)
        comms = tuple(sorted(c) for c in next(comms_generator))
    return comms