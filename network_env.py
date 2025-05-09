import random

import networkx as nx
import torch
from torch import Tensor, nn


class NetworkEnvironment(nn.Module):
    def __init__(
        self, num_of_nodes: int, transmission_time: int, k: int = 4, p: float = 0.1, packet_creation_prob: float = 0.2
    ) -> None:
        super().__init__()

        self.num_of_nodes = num_of_nodes
        self.transmission_time = transmission_time
        self.packet_creation_prob = packet_creation_prob
        self.k = k
        self.p = p
        self.network_graph: nx.Graph = nx.connected_watts_strogatz_graph(n=num_of_nodes, k=k, p=p)

        # Add edge attributes (tranmission capacity and betweenness)
        for u, v in self.network_graph.edges():
            self.network_graph[u][v]["capacity"] = random.randint(2, 10)  # noqa: S311
            self.network_graph[u][v]["remaining"] = self.network_graph[u][v]["capacity"]
        bet_centrality = nx.edge_betweenness_centrality(self.network_graph)
        for (u, v), val in bet_centrality.items():
            self.network_graph[u][v]["betweenness"] = val

        self.register_buffer("adjacency", torch.zeros(num_of_nodes, num_of_nodes, dtype=torch.bool), persistent=False)
        self.adjacency: Tensor = self.adjacency  # for type hinting
        for u, v in self.network_graph.edges():
            self.adjacency[u, v] = True
            self.adjacency[v, u] = True

        # num_of_nodes x max_num_of_neighbors===num_of_nodes
        self.register_buffer("queues", torch.zeros(num_of_nodes, num_of_nodes, dtype=torch.long), persistent=False)
        self.queues: Tensor = self.queues  # for type hinting
        self.queues_packets_destination = [[[] for _ in range(num_of_nodes)] for _ in range(num_of_nodes)]
        self.queues_packets_source = [[[] for _ in range(num_of_nodes)] for _ in range(num_of_nodes)]
        self.queues_packets_time = [[[] for _ in range(num_of_nodes)] for _ in range(num_of_nodes)]

        self.register_buffer(
            "in_transit_packets_source",
            torch.full((transmission_time + 1, num_of_nodes, num_of_nodes), -1, dtype=torch.long),
            persistent=False,
        )
        self.in_transit_packets_source: Tensor = self.in_transit_packets_source  # for type hinting

        self.register_buffer(
            "in_transit_packets_destination",
            torch.full((transmission_time + 1, num_of_nodes, num_of_nodes), -1, dtype=torch.long),
            persistent=False,
        )
        self.in_transit_packets_destination: Tensor = self.in_transit_packets_destination  # for type hinting

        self.register_buffer(
            "in_transit_packets_time",
            torch.full((transmission_time + 1, num_of_nodes, num_of_nodes), -1, dtype=torch.long),
            persistent=False,
        )
        self.in_transit_packets_time: Tensor = self.in_transit_packets_time  # for type hinting

        packet_prob = torch.full((self.num_of_nodes, self.num_of_nodes), packet_creation_prob / (self.num_of_nodes - 1))
        packet_prob.fill_diagonal_(1 - packet_creation_prob)
        self.register_buffer("packet_prob", packet_prob, persistent=False)
        self.packet_prob: Tensor = self.packet_prob  # for type hinting

        self.register_buffer("total_arrived_packets", torch.zeros(num_of_nodes, num_of_nodes), persistent=False)
        self.total_arrived_packets: Tensor = self.total_arrived_packets  # for type hinting

        self.register_buffer("total_arrived_packets_time", torch.zeros(num_of_nodes, num_of_nodes), persistent=False)
        self.total_arrived_packets_time: Tensor = self.total_arrived_packets_time  # for type hinting

    @property
    def device(self) -> torch.device:
        return self.adjacency.device

    def reset(self) -> None:
        for u, v in self.network_graph.edges():
            self.network_graph[u][v]["remaining"] = self.network_graph[u][v]["capacity"]
        self.queues.zero_()
        self.queues_packets_destination = [[[] for _ in range(self.num_of_nodes)] for _ in range(self.num_of_nodes)]
        self.queues_packets_source = [[[] for _ in range(self.num_of_nodes)] for _ in range(self.num_of_nodes)]
        self.queues_packets_time = [[[] for _ in range(self.num_of_nodes)] for _ in range(self.num_of_nodes)]
        self.in_transit_packets_source.fill_(-1)
        self.in_transit_packets_destination.fill_(-1)
        self.in_transit_packets_time.fill_(-1)
        self.total_arrived_packets.zero_()
        self.total_arrived_packets_time.zero_()

    def create_packets(self) -> Tensor:
        created_packets = torch.multinomial(self.packet_prob, num_samples=1).flatten()
        created_packets[created_packets == torch.arange(self.num_of_nodes, device=self.device)] = -1
        return created_packets

    def step(
        self,
        node_packets_dest: Tensor,
        action_dist: torch.distributions.Categorical,
        node_packets_src: Tensor = None,
        node_packets_time: Tensor = None,
    ) -> dict[str, Tensor]:
        # for each node with new packet, I must choose in which queue this packet should go
        # this actions tells me which queue to put the packet in
        actions = action_dist.sample()  # sample action from the distribution
        action_log_prob = action_dist.log_prob(actions)  # log prob of the sampled actions
        entropy = action_dist.entropy()

        src_nodes = torch.arange(self.num_of_nodes, device=self.device)

        # If destination is self-loop, set it to -1 (no packet)
        destinations = actions.clone()
        destinations[destinations == src_nodes] = -1
        # Select only nodes that generated packets (destinations != -1)
        node_with_new_packets = destinations >= 0
        src_nodes = src_nodes[node_with_new_packets]
        dst_nodes = destinations[node_with_new_packets]

        # Add the packet to the corresponding queue
        self.queues[src_nodes, dst_nodes] += 1

        for packet_final_dst, packet_src, packet_dst in zip(
            node_packets_dest[node_with_new_packets], src_nodes, dst_nodes, strict=False
        ):
            self.queues_packets_destination[packet_src][packet_dst].append(packet_final_dst)

        node_packets_src = node_packets_src[node_with_new_packets] if node_packets_src is not None else src_nodes
        for packet_initial_src, packet_src, packet_dst in zip(node_packets_src, src_nodes, dst_nodes, strict=False):
            self.queues_packets_source[packet_src][packet_dst].append(packet_initial_src)

        node_packets_time = (
            node_packets_time[node_with_new_packets] if node_packets_time is not None else torch.zeros_like(src_nodes)
        )
        for packet_time, packet_src, packet_dst in zip(node_packets_time, src_nodes, dst_nodes, strict=False):
            self.queues_packets_time[packet_src][packet_dst].append(packet_time)

        return {
            "action_log_prob": action_log_prob[node_with_new_packets],
            "action_entropy": entropy[node_with_new_packets],
        }

    def advance_time(self) -> dict[str, Tensor]:
        # For the arrival tensors, each row corresponds to a destination node for the packet while the columns are
        # the source nodes for the packet. The value at (i, j) is the destination/source of a packet that arrived at
        # node i from node j.

        # TRANSMIT
        ## Progress transit buffer (each step, packets get closer to destination)
        arrived_packets_source = self.in_transit_packets_source[0]
        arrived_packets_destination = self.in_transit_packets_destination[0]
        arrived_packets_time = self.in_transit_packets_time[0]

        self.in_transit_packets_source = torch.roll(self.in_transit_packets_source, shifts=-1, dims=0)
        self.in_transit_packets_destination = torch.roll(self.in_transit_packets_destination, shifts=-1, dims=0)
        self.in_transit_packets_time = torch.roll(self.in_transit_packets_time, shifts=-1, dims=0)
        self.in_transit_packets_source[-1].fill_(-1)  # Clear the last slice (new empty step)
        self.in_transit_packets_destination[-1].fill_(-1)  # Clear the last slice (new empty step)
        self.in_transit_packets_time[-1].fill_(-1)  # Clear the last slice (new empty step)

        # put a packet on the line, the packet should take T timesteps to cross the line
        # check for the packets that finished the line and arrrived in new nodes
        # those are now the "new_packets" for the next step
        ready_to_transmit = self.queues > 0
        self.queues[ready_to_transmit] -= 1

        # # Put packet at the last "step" (will arrive after T_TRANSMIT steps)
        # self.in_transit_packets[-1][ready_to_transmit] = True

        for src_node, queues in enumerate(self.queues_packets_destination):
            for dst_node, packets in enumerate(queues):
                # Count how many packets are currently on (src_node -> dst_node)
                num_in_flight = (self.in_transit_packets_destination[:, src_node, dst_node] != -1).sum()
                # Check if there are packets to transmit and there is enough capacity to transmit the packet
                if not packets or num_in_flight == self.network_graph[src_node][dst_node]["capacity"]:
                    continue

                # Transmit the packets
                packet_final_destination = packets.pop(0)
                self.in_transit_packets_destination[-1][src_node][dst_node] = packet_final_destination

        for src_node, queues in enumerate(self.queues_packets_source):
            for dst_node, packets in enumerate(queues):
                # Count how many packets are currently on (src_node -> dst_node)
                num_in_flight = (self.in_transit_packets_source[:, src_node, dst_node] != -1).sum()
                # Check if there are packets to transmit and there is enough capacity to transmit the packet
                if not packets or num_in_flight == self.network_graph[src_node][dst_node]["capacity"]:
                    # if not packets:
                    continue

                # Transmit the packets
                packet_initial_source = packets.pop(0)
                self.in_transit_packets_source[-1][src_node][dst_node] = packet_initial_source

        for src_node, queues in enumerate(self.queues_packets_time):
            for dst_node, packets in enumerate(queues):
                # Count how many packets are currently on (src_node -> dst_node)
                num_in_flight = (self.in_transit_packets_time[:, src_node, dst_node] != -1).sum()
                # Check if there are packets to transmit and there is enough capacity to transmit the packet
                if not packets or num_in_flight == self.network_graph[src_node][dst_node]["capacity"]:
                    continue

                # Calculate remaining capacity (only on the last for loop here to only subtract once)
                self.network_graph[src_node][dst_node]["remaining"] = (
                    self.network_graph[src_node][dst_node]["capacity"] - num_in_flight - 1
                )
                assert self.network_graph[src_node][dst_node]["remaining"] >= 0

                # Transmit the packets
                packet_time = packets.pop(0)
                self.in_transit_packets_time[-1][src_node][dst_node] = packet_time

        # ADVANCE TIME
        arrived_packets_time[arrived_packets_time != -1] += 1
        self.in_transit_packets_time[self.in_transit_packets_time != -1] += 1
        for src_node, queues in enumerate(self.queues_packets_time):
            for dst_node, packets in enumerate(queues):
                if not packets:
                    continue
                self.queues_packets_time[src_node][dst_node] = [time + 1 for time in packets]

        return {
            "destinations": arrived_packets_destination,
            "sources": arrived_packets_source,
            "times": arrived_packets_time,
        }

    def check_arrival(self, destinations: Tensor, sources: Tensor, times: Tensor) -> dict[str, Tensor]:
        """Check and filter arrived packets.

        Checks for packets that arrived at their final destinations, updates the total count, and removes them from the
        current arrival tensors.

        Args:
            destinations: Tensor (num_nodes, num_nodes). destinations[i,j] is the final destination of
                the packet arriving at node j from node i.
            sources: Tensor (num_nodes, num_nodes). sources[i,j] is the initial source of the packet
                arriving at node j from node i.
            times: Tensor (num_nodes, num_nodes). times[i,j] is the timesteps from creation of the
                packet arriving at node j from node i.

        Returns:
            A dictionary containing the updated 'destinations', and 'sources' tensors, with packets that reached their
            final destination removed.

        """
        arrived_destinations, arrived_sources, arrived_times = destinations, sources, times
        # Create a tensor representing the current node index for each column
        # Shape: (1, num_nodes) -> [[0, 1, 2, ..., num_nodes-1]]
        current_node_indices = torch.arange(self.num_of_nodes, device=self.device).unsqueeze(0)

        # Mask for packets where the current node (j) matches their final destination
        is_final_destination = arrived_destinations == current_node_indices

        # Combine masks: True where a packet arrived AND current node is its final destination
        arrived_at_final_destination_mask = (arrived_destinations != -1) & is_final_destination

        # Check if any packets reached their final destination in this step
        if not arrived_at_final_destination_mask.any():
            return {"destinations": arrived_destinations, "sources": arrived_sources, "times": arrived_times}

        # --- Update total_arrived_packets ---
        # Get the initial sources, final destinations, and counts of the packets
        # that successfully reached their final destination.
        total_time = arrived_times[arrived_at_final_destination_mask]
        initial_source = arrived_sources[arrived_at_final_destination_mask]
        final_destination = arrived_destinations[arrived_at_final_destination_mask]

        # Ensure indices are valid before updating total_arrived_packets
        assert (initial_source >= 0).all() and (final_destination >= 0).all() and (total_time >= 0).all()

        valid_indices_mask = (initial_source >= 0) & (final_destination >= 0) & (total_time >= 0)
        if valid_indices_mask.any():
            # Use index_put_ with accumulate=True to handle multiple packets arriving
            # at the same final destination from the same initial source simultaneously.
            # Filter potentially invalid indices (e.g., from placeholders if they weren't filtered before)
            valid_total_time = total_time[valid_indices_mask]
            valid_initial_source = initial_source[valid_indices_mask]
            valid_final_destination = final_destination[valid_indices_mask]
            self.total_arrived_packets[valid_initial_source, valid_final_destination] += 1
            self.total_arrived_packets_time[valid_initial_source, valid_final_destination] += valid_total_time

        # --- Remove successfully arrived packets from current tensors ---
        # Create copies to avoid modifying the original tensors if they are needed elsewhere
        updated_arrived_destination = arrived_destinations.clone()
        updated_arrived_source = arrived_sources.clone()
        updated_arrived_time = arrived_times.clone()

        # Set entries for successfully arrived packets to 0 or placeholder
        updated_arrived_destination[arrived_at_final_destination_mask] = -1
        updated_arrived_source[arrived_at_final_destination_mask] = -1
        updated_arrived_time[arrived_at_final_destination_mask] = -1

        return {
            "destinations": updated_arrived_destination,
            "sources": updated_arrived_source,
            "times": updated_arrived_time,
        }
