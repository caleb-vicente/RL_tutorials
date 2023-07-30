import torch
import torch.nn as nn


class Structure2Vec(nn.Module):

    def __init__(self, nodes_cost, num_features, embed_dim=64, num_updates=4, num_nodes=50):
        super(Structure2Vec, self).__init__()

        self.nodes_features = torch.zeros(num_nodes, num_features)  # Size is: [num_nodes, num_features]
        self.nodes_cost = torch.tensor(nodes_cost)  # Size is: [num_nodes, num_nodes]
        # TODO: Make sure that nodes features, cost and embed has the grad True or False depending on the necessity of the problem
        self.embed_dim = embed_dim
        self.num_updates = num_updates
        self.num_nodes = num_nodes
        self.tensor_q = torch.zeros((self.num_nodes, 1))

        self.matrix_embeddings = torch.zeros(self.num_nodes, self.embed_dim)

        # Nodes features must contain as many rows as number of nodes are in the graph. Number of features might vary.
        # Typically will be a single feature. 1 if the node has been traveled and 0 otherwise
        if not self.nodes_features.size()[0] == self.num_nodes:
            raise ValueError(
                f"Nodes Features size {tuple(self.nodes_features.size())} does not match the number of nodes: {num_nodes}")

        if not self.nodes_cost.size() == (num_nodes, num_nodes):
            raise ValueError(
                f"Nodes cost size {tuple(self.nodes_cost.size())} does not match the necessary: {self.num_nodes}")

        # Embedding neural network
        self.linear1 = nn.Linear(1, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear3 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear4 = nn.Linear(1, self.embed_dim)
        self.relu = nn.ReLU()
        self.relu_cost = nn.ReLU()

        # Q-network
        self.linear5 = nn.Linear(2 * self.embed_dim, 1)
        self.linear6 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear7 = nn.Linear(self.embed_dim, self.embed_dim)
        self.reluQ = nn.ReLU()

    def forward(self, nodes_features):

        # Transform ndarray state to torch tensor
        self.nodes_features = nodes_features.T

        # Update all embeddings
        for i in range(self.num_updates):
            self.compute_all_embeddings()

        # Compute all Qs
        for v in range(self.num_nodes):
            q = torch.tensor(self.compute_Q(v)).unsqueeze(0)  # convert Q value to tensor and add an extra dimension
            self.tensor_q[v] = q  # concatenate the new Q value to the existing tensor

        return self.tensor_q

    def compute_all_embeddings(self):

        new_matrix_embeddings = self.matrix_embeddings.clone()

        for node_id in range(self.num_nodes):

            vector1 = self.linear1(self.nodes_features[node_id, :])
            vector2 = self.linear2(self.matrix_embeddings.sum(dim=0))
            vector3 = self.relu_cost(self.linear4(self.nodes_cost[:, node_id].unsqueeze(1).to(torch.float32))).sum(dim=0)  # TODO: relu works in the proper dimension?, check dims

            new_matrix_embeddings[node_id, :] = self.relu(vector1 + vector2 + vector3)

        self.matrix_embeddings.data = new_matrix_embeddings.data

    def compute_Q(self, node_id):
        """
        This function only calculates the Q for a given state and a given vector. This means that this function must be
        call several times each iteration to obtain the values of all remaining possible states.

        :return: scalar with the value of q for v given the current status of the network
        """
        vector1 = self.linear6(self.matrix_embeddings.sum(dim=0))  # [1 x p]
        vector2 = self.linear7(self.matrix_embeddings[node_id, :])  # [1 x p]
        concat_vector = torch.cat((vector1, vector2), dim=0)
        q_v = self.linear5(self.reluQ(concat_vector))

        return q_v





