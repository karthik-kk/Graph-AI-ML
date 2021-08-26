# Link Prediction with Graph Convolutional Networks(GCNs) using Deep Graph Library(DGL)

## Dataset:
- UIDs represent nodes (Person nodes).
- Experience_ID/Experience represent nodes (Experience nodes).
- Post_Emotion represent the edges between the nodes UID and Experience.
- Post_Valence represent the edge weights of the emotion between the nodes.
- [TBD: More info on Valence ?? Can valence be considered for edge weights?]

## Objective:
- To Train a GCN for Link prediction.
- The input features will be the edge weights which contains the emotion valence scores which will be translated to either positive or negative values for edges.
- Output will be the predicted scores (either +ve or -ve) for each UID with the corresponding experience.
