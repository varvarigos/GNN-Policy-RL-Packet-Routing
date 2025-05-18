import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical
from torch_geometric.nn import NNConv

from network_env import NetworkEnvironment


def graph_to_data(environment: NetworkEnvironment, new_packets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    G, queue_lengths = environment.network_graph, environment.queues

    device = queue_lengths.device

    edge_index, edge_attr = [], []
    for u, v, data in G.edges(data=True):
        edge_index.extend([[u, v], [v, u]])
        edge_attr.extend([[ data["remaining"] / data["capacity"]]] * 2)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)

    adjacency = environment.adjacency

    x = torch.stack(
        [
            torch.cat(
                [
                    torch.tensor([G.degree[node]]).to(new_packets),
                    queue_lengths[node].masked_fill(adjacency[node] == 0, -1),
                    new_packets[node].unsqueeze(0)
                ]
            )
            for node in G.nodes()
        ]
    ).to(dtype=torch.float)

    return x, edge_index, edge_attr


class GNNPolicy(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_of_nodes: int) -> None:
        super().__init__()
        node_dim += num_of_nodes
        nn1 = nn.Sequential(
            nn.Linear(edge_dim, node_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(node_dim * hidden_dim, node_dim * hidden_dim),
        )
        self.conv1 = NNConv(node_dim, hidden_dim, aggr="mean", nn=nn1)

        nn2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim ** 2),
        )
        self.conv2 = NNConv(hidden_dim, hidden_dim, aggr="mean", nn=nn2)
        self.output = nn.Linear(hidden_dim, num_of_nodes)  # Predict logits for next node (not pairwise manually)

    def forward(self, new_packets: Tensor, environment: NetworkEnvironment) -> torch.distributions.Categorical:
        x, edge_index, edge_attr = graph_to_data(environment, new_packets)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        logits: Tensor = self.output(x)  # Now directly output node logits

        adjacency = environment.adjacency
        # logits : num_of_nodes x max_num_of_neighbors===num_of_nodes
        masked_logits = logits.masked_fill(adjacency == 0, -torch.inf)  # mask invalid moves
        probs = F.softmax(masked_logits, dim=1)
        assert torch.allclose(probs, probs * (adjacency != 0))
        # for nodes that did not create a packet, I want the prob to be 1 for self-loop on the index that is the node
        # that created the packet
        return_probs = probs.clone()
        nodes_that_created_packets = new_packets != -1
        return_probs[~nodes_that_created_packets, :] = 0.0
        # pad an extra column of zeros to the end of the probs matrix
        return_probs[~nodes_that_created_packets, ~nodes_that_created_packets] = 1.0

        return Categorical(return_probs)
