import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm 
import re
from collections import defaultdict
import seaborn as sns
from sklearn.preprocessing import normalize
import sklearn
import networkx as nx
import community.community_louvain as community_louvain
import igraph as ig
from model import *
from collections import OrderedDict
import logging
import faiss
import time

# Initializing the Training Dataframe from github
PATH='./Datasets/train_embeddings.csv'
trainDF = pd.read_csv(PATH)
message_embeddings_size = 512  # size of each message embedding

all_embeddings=[]
for i in range(len(trainDF)):
    embeddings=[]
    for j in range(1,513):
        embeddings.append(trainDF.iloc[i][j])
    all_embeddings.append(embeddings)

messageEmbeddings=np.asarray(all_embeddings, dtype=np.float32)

for i in range(len(messageEmbeddings)):
        messageEmbeddings[i] /= np.linalg.norm(messageEmbeddings[i])


"""# Utility Functions"""

def neighborhoodSearch(emb,thresh,k_neighbors):
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)
    sim, I = index.search(emb, k_neighbors)
    pred_index=[]
    pred_sim=[]
    for i in range(emb.shape[0]):
        d = len(sim[i])
        left = 0
        right = d - 1
        while left <= right:
            mid = (left + right) // 2
            if sim[i][mid] < thresh:
                right = mid - 1
            else:
                left = mid + 1
        cut_index=right+1
        pred_index.append(I[i][:(cut_index)])
        pred_sim.append(sim[i][:(cut_index)])
        
    return pred_index,pred_sim
    
def blendNeighborhood(emb, adjacencyList,adjacencySimilarities):
    new_emb = emb.copy()
    for i in range(emb.shape[0]):
        cur_emb = emb[adjacencyList[i]]
        weights = np.expand_dims(adjacencySimilarities[i], 1)
        new_emb[i] = (cur_emb * weights).sum(axis=0)
    new_emb = sklearn.preprocessing.normalize(new_emb, axis=1)
    return new_emb

def iterativeNeighborhoodBlending(emb, num_iter=2,onlyNebr=10,threshes=None,blend=True):
    if threshes==None:
        for _ in range(num_iter):
            adjacencyList,adjacencySimilarities = neighborhoodSearch(emb, 0,onlyNebr)
            if blend:
                emb = blendNeighborhood(emb, adjacencyList, adjacencySimilarities)
        return adjacencyList,adjacencySimilarities
    else:
        for thresh in threshes:
            adjacencyList,adjacencySimilarities = neighborhoodSearch(emb, thresh,32)
            if blend:
                emb = blendNeighborhood(emb, adjacencyList, adjacencySimilarities)
        return adjacencyList,adjacencySimilarities

def createGraph(adjacencyList,adjacencySimilarities,messageEmbeddings):
    adjacencyDict = {i: adjacencyList[i] for i in range(0, len(adjacencyList))}
    graph = nx.from_dict_of_lists(adjacencyDict)
    for i in range(messageEmbeddings.shape[0]):
        for j in range(len(adjacencyList[i])):
            graph[i][adjacencyList[i][j]]['weight']=adjacencySimilarities[i][j]
    return graph

def getCommunitiesInGraph(graph,method):
    memberships = {}
    if method == "louvain":
        memberships = community_louvain.best_partition(graph)
    elif method=='leidenalg':        
        r=graph.edges.data('weight')
        G2 = ig.Graph.TupleList(r, directed=False,weights=True)
        weights = np.array(G2.es["weight"]).astype(np.float64)
        results=leidenalg.find_partition(G2, leidenalg.ModularityVertexPartition,n_iterations=-1,weights=weights)
        memberships={}
        communities = list(results)
        for i, single_community in enumerate(list(communities)):
            for member in single_community:
                memberships[int(member)] = i
    elif method == "girvan_newman":
        result = nx.algorithms.community.girvan_newman(graph)
        communities = next(result)
        for i, single_community in enumerate(list(communities)):
            for member in single_community:
                memberships[int(member)] = i
    memberships = {k: v for k, v in sorted(list(memberships.items()), key=lambda x: x[0])}
    return memberships

def refineClustering(logger,memberships,messageEmbeddings,currentMethod,threshes=[0.5]):
    clusterCenter={}
    clusterSize={}
    for i in range(len(memberships)):
        if memberships[i] not in clusterCenter:
            clusterCenter[memberships[i]]=np.array(len(messageEmbeddings[i])*[0],dtype='float32')
        clusterCenter[memberships[i]]+=messageEmbeddings[i]
        if memberships[i] not in clusterSize:
            clusterSize[memberships[i]]=0
        clusterSize[memberships[i]]+=1
    for i in range(len(clusterCenter)):
        clusterCenter[i]/=clusterSize[i]
    clusterCenterEmbeddings=np.array(list(clusterCenter.values()),dtype='float32')
    for i in range(len(clusterCenterEmbeddings)):
        clusterCenterEmbeddings[i] /= np.linalg.norm(clusterCenterEmbeddings[i])

    centerAdjacencyList,centerAdjacencySimilarities=iterativeNeighborhoodBlending(clusterCenterEmbeddings,blend=True,threshes=threshes)
    graphCenter=createGraph(centerAdjacencyList,centerAdjacencySimilarities,clusterCenterEmbeddings)
    for i in range(clusterCenterEmbeddings.shape[0]):
        for j in range(len(centerAdjacencyList[i])):
            graphCenter[i][centerAdjacencyList[i][j]]['weight']=centerAdjacencySimilarities[i][j]

    membershipsCenter = list(getCommunitiesInGraph(graphCenter, method=currentMethod).values())

    newClusters=max(membershipsCenter)+1
    logger.info('Number of Cluster After Merging: '+str(max(membershipsCenter)+1))

    groupCenterCluster={}
    for i in range(len(membershipsCenter)):
        if membershipsCenter[i] not in groupCenterCluster:
            groupCenterCluster[membershipsCenter[i]]=[]
        groupCenterCluster[membershipsCenter[i]].append(i)
    mapOldToNewCluster={}
    clusterNumber=0
    for mergedClusters in groupCenterCluster.values():
        for cluster in mergedClusters:
            mapOldToNewCluster[cluster]=clusterNumber
        clusterNumber+=1
    newMemberships=memberships.copy()
    for i in range(len(newMemberships)):
        newMemberships[i]=mapOldToNewCluster[newMemberships[i]]
    return newMemberships

"""# Driver Part"""

logging.basicConfig(filename="./Results/Clustering/Clustering.log",format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

algo='louvain'


threshes1=[[0.9,0.92],[0.9,0.94],[0.9,0.96],[0.9,0.98],[0.92,0.94],[0.92,0.96],[0.92,0.98],[0.94,0.96],[0.94,0.98],[0.96,0.98]]
threshes2=[[0.9],[0.92],[0.94],[0.96],[0.98]]

for i,thresh1 in enumerate(threshes1):
    for j,thresh2 in enumerate(threshes2):
        logger.info(f'NEW EPOCH: Running {algo} with initial threshold as {thresh1} and refinement threshold as {thresh2}')
        adjacencyList,adjacencySimilarities=iterativeNeighborhoodBlending(messageEmbeddings,threshes=thresh1)

        graph=createGraph(adjacencyList,adjacencySimilarities,messageEmbeddings)
        memberships=getCommunitiesInGraph(graph,algo)
        logger.info('Initial Number of Clusters: '+str(max(list(memberships.values()))+1))
        trainDF['raw_predicted_community']=list(memberships.values())
        values=[1000,500,250,100,50]
        for value in values:
            logger.info('Number of Cluster with size atleast '+str(value)+': '+str(np.sum(trainDF['raw_predicted_community'].value_counts()>=value)))
        logger.info('Starting Refinement of Clusters')
        newMemberships=refineClustering(logger,memberships,messageEmbeddings,algo,thresh2)
        trainDF['refined_predicted_community']=list(newMemberships.values())
        for value in values:
            logger.info('Number of Cluster with size atleast '+str(value)+': '+str(np.sum(trainDF['refined_predicted_community'].value_counts()>=value)))

        date=time.strftime("%d%m")
        tt=time.strftime("%H%M")
        fileName=f'./Results/Clustering/clustering_{i}_{j}.xlsx'
        trainDF.to_excel(fileName)
