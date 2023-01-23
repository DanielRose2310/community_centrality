import math
import networkx as nx
import numpy as np
from clustering import clustering_algo
from utils import angle_between
from halo import Halo
np.seterr(all="ignore")

spinner = Halo(text='Loading', spinner='dots')

def comm_centrality(_G,clustering='gn',weight=None):
    spinner.start()
    mm = nx.modularity_matrix(_G,weight='weight')
    comms = clustering_algo(_G,clustering,weight)
    eigs = np.linalg.eig(mm)
    eig_vals = eigs[0]
    eig_vecs = eigs[1]
    
    pos_vals = []
    pos_vecs = []
    
    for i,val in enumerate(eig_vals):
        if val>0:
            pos_vals.append(val)
            pos_vecs.append(np.array(eig_vecs[i])[0])
            
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
            
            comp = (math.sqrt(pos_val) * pos_vecs[j][i]) ** 2
            
            node_vector.append(comp)

        vector_normed = node_vector / np.linalg.norm(node_vector)
        node_vectors[node_name] = vector_normed
        xi = np.sum(vector_normed) 
        contribs_dict[node_name] = xi
        contribs.append(xi)

    comm_vecs = dict()
    cluster_dict = dict()
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

    solidarity_dict = dict()
    solidarity_list = []
    for node_n,node_v in node_vectors.items():
        for comm_i,comm_nodes in enumerate(comms):
                if node_n in comm_nodes:
                    v1 = np.array(node_v)[0]
                    v2 = np.array(comm_vecs_k[comm_i])[0]
        
                    angle = angle_between(v1,v2)
                    solidarity = abs(math.cos(angle))
                    solidarity_dict[node_n] = solidarity 
                    solidarity_list.append(abs(solidarity))
                    

    
    solidarity_list = np.array(solidarity_list)
    
    spinner.stop()

    return contribs_dict,cluster_dict,solidarity_dict