community_centrality
Eigenspectral community centrality algorithm based for network analysis

  from comm_cent import community_centrality
  
  contrib,clusters,solidarity = community_centrality(G,clustering,weight)
arguments:

G (networkx graph): graph data

clusters (string) (default='gn'): clustering algorithm to use

currently supports 'gn' (Girvan-Newman) and 'louvain' (Louvain)
weight (int) (default=None): weight parameter

returns:

contrib (dictionary): each node's relative contribution to modularity

clusters (dictionary): clustering of each node

solidarity (dictionary): each node alignment with its respective community

Implementation of: Newman ME. Finding community structure in networks using the eigenvectors of matrices. Phys Rev E Stat Nonlin Soft Matter Phys. 2006 Sep;74(3 Pt 2):036104. doi: 10.1103/PhysRevE.74.036104. Epub 2006 Sep 11. PMID: 17025705.