import math
import networkx as nx
import numpy as np
from clustering import clustering_algo
from utils import angle_between
from halo import Halo
np.seterr(all="ignore")

spinner = Halo(text='Loading', spinner='dots')

def community_centrality(_G,clustering='gn',weight=None):
    spinner.start()
    mm = nx.modularity_matrix(_G,weight=weight) #create modularity matrix of data
    comms = clustering_algo(_G,clustering,weight) #deploy clustering algorithm
    eig_vals,eig_vecs = np.linalg.eig(mm) #fetch eigenvalues & eigenvectors
    
    pos_vals = []
    pos_vecs = []
    
    for i,val in enumerate(eig_vals):
        if val>0:
            pos_vals.append(val)
            pos_vecs.append(np.array(eig_vecs[i]).flatten().real) #for tidiness, keep all positive vals & vecs in a seperate matrix
            
    if len(pos_vals) < 1:
        print('No good eigenvalues')
        return 
    
    comm_vecs = dict()
    contribs = []
    contribs_dict = dict()
    node_vectors = dict()
    for i,node in enumerate(_G.nodes):
        node_vector = []
        node_name = node
        
        for j,pos_val in enumerate(pos_vals):
            
            comp = (math.sqrt(pos_val) * pos_vecs[j][i]) ** 2 #get component from pos eigenvectors of index
            
            node_vector.append(comp)

        vector_normed = node_vector / np.linalg.norm(node_vector) #normalize component vector
        node_vectors[node_name] = vector_normed
        xi = np.sum(vector_normed) 
        contribs_dict[node_name] = xi
        contribs.append(xi)

    comm_vecs = dict()
    cluster_dict = dict()
    
    # calculate and sum community vectors
    for node_n,node_v in node_vectors.items():
        for comm_i,comm_nodes in enumerate(comms):
            if node_n in comm_nodes:
                cluster_dict[node_n]  = comm_i
                if comm_i not in comm_vecs.keys():
                    comm_vecs[comm_i] = []      
                comm_vecs[comm_i].append(node_v)

    comm_vecs_k = dict()
    for comm_i,comm_nodes in enumerate(comms):
        comm_vecs_k[comm_i] = np.add.reduce(comm_vecs[comm_i])  
    
    #calculate diffs of node vectors from comm vectors and cast as cos of angle
    solidarity_dict = dict()
    for node_n,node_v in node_vectors.items():
        for comm_i,comm_nodes in enumerate(comms):
                if node_n in comm_nodes:
                    v1 = np.array(node_v)
                    v2 = np.array(comm_vecs_k[comm_i])
                    
                    angle = angle_between(v1,v2)
                    solidarity = abs(math.cos(angle))
                    solidarity_dict[node_n] = solidarity 
    
    spinner.stop()

    return contribs_dict,cluster_dict,solidarity_dict
