import numpy as np
import csv

def import_facebook_data(filename):
    file = open(filename,"r")
    sdata = file.read().splitlines()
    data = [list(map(int,i.split())) for i in sdata]
    d = {}
    ndata = list()
    for i in data:
        if tuple(i) not in d and tuple(list(reversed(i))) not in d:
            d[tuple(i)] = 0
            ndata.append(i)
    return np.unique(np.array(ndata),axis=0)


def import_bitcoin_data(file_path):
    data = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = np.array(data)
    return np.array(data[:,[0,1]]).astype(int)

def spectralDecomp_OneIter(data):
    ndata = np.unique(data.flatten())
    num_vert = ndata.shape[0]
    vert_mp = {}
    rev_mp = {}
    k = 0
    for i in ndata:
        vert_mp[i] = k
        rev_mp[k] = i
        k += 1
    adj_mat = np.zeros((ndata.shape[0],ndata.shape[0]),dtype=int)
    for i in data:
        adj_mat[vert_mp[i[0]],vert_mp[i[1]]] = 1
        adj_mat[vert_mp[i[1]],vert_mp[i[0]]] = 1
    
    
    deg = np.diag(adj_mat.sum(axis=0))
    L = deg-adj_mat
    
    eig_val,eig_vec = np.linalg.eigh(L)
    eig = [(i,j) for i,j in zip(eig_val,eig_vec.T)]
    
    d = sorted(eig,key= lambda x: x[0])
    for i in d:
        if i[0] > 0.00001:
            fiedler_vec = i[1]
            break
    lst = sorted(list(enumerate(fiedler_vec)),key=lambda x: x[1])
    el = list(zip(*lst))
    part = np.where(np.array(el[1])>=0)[0].shape[0]
    
    com = 0
    if part>num_vert//2:
        com = 1
    graph_partition = np.zeros((num_vert,2),dtype=int)
    
    for i in lst:
        if i[1]>0:
            graph_partition[i[0]][0] = rev_mp[i[0]]
            graph_partition[i[0]][1] = com
        else:
            graph_partition[i[0]][0] = rev_mp[i[0]]
            graph_partition[i[0]][1] = com^1
    return fiedler_vec,adj_mat,graph_partition

def spectralDecomposition(data):
    num_vert = data.max()
    adj_mat = np.zeros((num_vert+1,num_vert+1),dtype=int)
    for i in data:
        adj_mat[i[0],i[1]] = 1
        adj_mat[i[1],i[0]] = 1
    
    deg = np.diag(adj_mat.sum(axis=0))
    L = deg-adj_mat
    
    deg_half = 1.0/np.sqrt(adj_mat.sum(axis=0))
    deg_half[np.isinf(deg_half)] = 0
    deg_h = np.diag(deg_half)
    norm_L = deg_h@(L@deg_h)
    
    eig_val,eig_vec = np.linalg.eigh(norm_L)
    d = {}
    for i,j in zip(eig_val,eig_vec.T):
        d[i] = j.T
    d = dict(sorted(d.items()))
    
    eig = np.sort(eig_val)
    k = 0
    m = -1
    for i in range(1,eig.shape[0]):
        if eig[i]-eig[i-1]>m:
            m = eig[i]-eig[i-1]
            k = i
    graph_partition = np.zeros((num_vert+1,2),dtype=int)
    for i in range(num_vert+1):
        graph_partition[i][0] = i
    clst_size = 4039
    for i in range(7):
        print("Community : "+str(i))
        fiedler_vec,adj_mat,graph_tmp = spectralDecomp_OneIter(data)
        ndata = []
        for j in range(adj_mat.shape[0]):
            if graph_tmp[j][1] == 0:
                continue
            for l in range(adj_mat.shape[0]):
                if adj_mat[j][l]==1 and graph_tmp[l][1] != 0:
                    ndata.append([graph_tmp[j][0],graph_tmp[l][0]])
        data = np.array(ndata)
        for j in graph_tmp:
            if j[1]==0:
                continue
            graph_partition[j[0]][1] = i+1
        clst_size = np.unique(data.flatten()).shape[0]
        i += 1
    return graph_partition

def createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb):
    num_vert = nodes_connectivity_list_fb.max()
    adj_mat = np.zeros((num_vert+1,num_vert+1),dtype=int)
    for i in nodes_connectivity_list_fb:
        adj_mat[i[0],i[1]] = 1
        adj_mat[i[1],i[0]] = 1
    
    idx = np.array(sorted(graph_partition_fb,key=lambda x:x[1])).T[0]
    ix = np.ix_(idx,idx)
    n_adj_mat = adj_mat[ix]
    return n_adj_mat

def louvain_one_iter(nodes_connectivity_list_fb):
    return
if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
