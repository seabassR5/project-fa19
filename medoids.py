import sys
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
#TODO import an use networkX, my machine isnt cooperating atm

def kMedoids(D, k, tmax=100):
    """
    this func is taken from this reaserch pap:
    https://www.researchgate.net/publication/272351873_NumPy_SciPy_Recipes_for_Data_Science_k-Medoids_Clustering
    basically its kmeans but using distances, not xy coors
    
    inputs: 

    """
    # determine dimensions of distance matrix D
    m, n = D.shape
    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))
    # create a copy of the array of medoid indices
    Mnew = np.copy(M)
    # initialize a dictionary to represent clusters
    C = {}
    for t in xrange(tmax):
    # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C
## running into issues because i have anaconda3, and python 3.7 installed and the associations in windows are fkd


##"""TODO """
## input the graph adjacency matrix into networkx func ne
## documentation link: 
## floyd_warshall_numpy(G, nodelist=None, weight='weight')
## params
## G (NetworkX graph) –
## nodelist (list, optional) – The rows and columns are ordered by the nodes in nodelist. If nodelist is None then the ordering is produced by G.nodes().
## weight (string, optional (default= ‘weight’)) – Edge data key corresponding to the edge weight.
##
## distance (NumPy matrix) – A matrix of shortest path distances between nodes. If there is no path between to nodes the corresponding matrix entry will be Inf.

