
import numpy as np
import json
import hashlib
from datetime import datetime

class HyperFractalMemory:
    def __init__(self):
        self.memory = {}
        self.timeline = []
        self.temporal_nodes = {}

    def _generate_hash(self, data):
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def store_memory(self, key, value, emotional_weight=0.5):
        timestamp = datetime.utcnow().isoformat()
        hashed_key = self._generate_hash({"key": key, "timestamp": timestamp})
        self.memory[hashed_key] = {
            "value": value,
            "timestamp": timestamp,
            "emotional_weight": emotional_weight,
            "connections": []
        }
        self.timeline.append(hashed_key)
        return hashed_key

    def link_memories(self, key1, key2):
        if key1 in self.memory and key2 in self.memory:
            self.memory[key1]["connections"].append(key2)
            self.memory[key2]["connections"].append(key1)

    def set_temporal_node(self, label, reference_point):
        if reference_point in self.memory:
            self.temporal_nodes[label] = reference_point

    def retrieve_memory(self, key):
        return self.memory.get(key, "Memory not found")

    def analyze_timeline(self):
        return {
            "total_memories": len(self.memory),
            "first_entry": self.memory[self.timeline[0]] if self.timeline else None,
            "latest_entry": self.memory[self.timeline[-1]] if self.timeline else None,
            "temporal_nodes": self.temporal_nodes
        }

    def vector_embedding(self, key, embedding_vector):
        if key in self.memory:
            self.memory[key]["embedding"] = embedding_vector

    def decay(self, threshold=0.2):
        keys_to_remove = [k for k, v in self.memory.items() if v.get("emotional_weight", 0) < threshold]
        for k in keys_to_remove:
            del self.memory[k]
            if k in self.timeline:
                self.timeline.remove(k)

    def visualize_memory_graph(self):
        import networkx as nx
        import plotly.graph_objects as go

        G = nx.Graph()
        for key, data in self.memory.items():
            G.add_node(key, label=data["value"], weight=data.get("emotional_weight", 0.5))
            for connection in data["connections"]:
                G.add_edge(key, connection)

        pos = nx.spring_layout(G, seed=42)
        node_trace = go.Scatter(
            x=[pos[k][0] for k in G.nodes()],
            y=[pos[k][1] for k in G.nodes()],
            text=[f"{k}: {G.nodes[k]['label']}<br>Emotional Weight: {G.nodes[k]['weight']:.2f}" for k in G.nodes()],
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=[20 * G.nodes[k]['weight'] for k in G.nodes()],
                color=[G.nodes[k]['weight'] for k in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Emotional Weight')
            )
        )

        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                mode='lines'
            ))

        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(title="Victor's HyperFractalMemory Graph", showlegend=False)
        fig.show()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
