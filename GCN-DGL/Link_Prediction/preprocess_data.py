#!/usr/bin/env python
# coding: utf-8
# Author: Karthik Krishnamurthy

# ## Load graph data from CSV
# * The `nodes.csv` stores every user and their attributes.
# * The `edges.csv` stores the interactions between the user and the experience.

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython import get_ipython
plt.show(block=True)
from IPython.display import HTML

import pandas as pd

nodes_data = pd.read_csv('data/nodes.csv')
print(nodes_data)


edges_data = pd.read_csv('data/edges.csv')
print(edges_data)


# Construct a graph where each node is a user/experience and each edge represents their interactions. 
# `'Src'` and `'Dst'` columns from the `edges.csv` table.
import dgl

src = edges_data['Src'].to_numpy()
dst = edges_data['Dst'].to_numpy()

# Create a DGL graph from a pair of numpy arrays
g = dgl.graph((src, dst))

# Print the graph - to see number of nodes and edges.
print(g)  #This is a DGL Graph

# Visualization: Convert DGL graph to a `networkx` graph for visualization.
import networkx as nx

nx_g = g.to_networkx().to_undirected()
# Kamada-Kawaii layout template for formatting graphs for visualization.
pos = nx.kamada_kawai_layout(nx_g)
color_map = []
for node in nx_g:
    if node < 10:
        color_map.append('lightblue')
    else: 
        color_map.append('orange')
#nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
nx.draw(nx_g, pos, with_labels=True, node_color=color_map)
plt.savefig('user_experience_emotion_synthetic_graph.png', bbox_inches='tight', transparent=True)

# ## Query graph structures
# Printing the # of nodes and edges in this graph.

print('#Nodes', g.number_of_nodes())
print('#Edges', g.number_of_edges())


# Graph Queries.
# Get the in-degree of node 0:
g.in_degree(0)
print('#of Edges for Node 0: ', g.in_degree(0))

# ## Load node and edge features
# DGL only accepts inputs in tensors.
# * For categorical attributes (e.g. experience, sentiment), convert them to integers or one-hot encoding.

import torch
import torch.nn.functional as F

# The "Experience" column represents positive or negative valence.

experience = nodes_data['Experience'].to_list()
# Convert to categorical integer values with 0 for 'Negative', 1 for 'Positive'.
experience = torch.tensor([c == 'Positive' for c in experience]).long()
print('KK:::', experience)
# We can also convert it to one-hot encoding.
exp_onehot = F.one_hot(experience)
print(exp_onehot)

# Use `g.ndata` like a normal dictionary
g.ndata.update({'experience' : experience, 'exp_onehot' : exp_onehot})
print(g)


# Feeding edge features to a DGL graph.
# Get edge features from the DataFrame and feed it to graph.
edge_weight = torch.tensor(edges_data['Weight'].to_numpy())
# Similarly, use `g.edata` for getting/setting edge features.
g.edata['weight'] = edge_weight
print(g)