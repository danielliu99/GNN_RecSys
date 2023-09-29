# Graph Neural Network (GNN) for User-Movie Recommendation 

## From Collaborative Filtering to GNN 
Collaborative Filtering (CF) is a classic recommender system. It utilizes the intuitive idea that if two users share some common interests in the historical data, then they are likely to have other common interests to be discovered. Users and items (movies) naturally become a bipartite graph data structure, in which users and items are nodes, the interactions (a user likes a movie or two users are friends) between them are edges. 

Graph Neural Networks (GNN) utilizes the similar idea, while makes the model more powerful. In GNN, nodes are represented as embeddings with much more rich features, a node can learn information from nodes that are several hops away, and, as a deep learning method, GNN can capture non-linear relationships between nodes. 

## The Message Passing Scheme 

