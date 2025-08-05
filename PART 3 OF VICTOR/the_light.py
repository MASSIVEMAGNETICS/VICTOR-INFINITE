import numpy as np
import random
import copy
import math
import pickle # Add pickle for save/load state
import os # Make sure os is imported

class FlowerOfLifeMesh3D:
    def __init__(self, depth=3, radius=1.0, base_nodes=37, compute_adjacency_for_base=True, num_neighbors=6):
        self.depth, self.radius, self.base_nodes_count = depth, radius, base_nodes
        self.nodes = {}  # Store node_id: {coords, type, depth}
        self.adjacency = {} # Store node_id: [neighbor_ids]
        self.num_neighbors_setting = num_neighbors # Used for generating adjacency for base layer

        if self.base_nodes_count == 1:
            self._add_node(0, (0,0,0), "primary", 0)
        elif self.base_nodes_count == 7: # Standard 2D Flower of Life base
            self._generate_2d_fol_base(depth=0)
        elif self.base_nodes_count == 19: # Extended 2D Flower of Life base
             self._generate_2d_fol_base(depth=0, rings=2) # Assumes rings=1 for 7, rings=2 for 19
        elif self.base_nodes_count == 37: # Further extended 2D Flower of Life base
            self._generate_2d_fol_base(depth=0, rings=3)
        else: # Default to sphere packing if not a standard FoL base node count
            self._generate_sphere_packing_base(self.base_nodes_count)
        
        current_base_nodes = list(self.nodes.keys()) # Nodes created by base generation

        if compute_adjacency_for_base and self.base_nodes_count > 1:
            self._compute_adjacency_for_layer(current_base_nodes, num_neighbors=self.num_neighbors_setting)

        if depth > 0: # Build higher-dimensional layers if depth > 0
            self._construct_layers(current_base_nodes, depth)
            
    def _add_node(self, node_id, coords, node_type="primary", depth_level=0, is_new_layer_node=False):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "coords": np.array(coords), "type": node_type, "depth": depth_level, "is_new_layer_node": is_new_layer_node}
            self.adjacency[node_id] = []
            return True
        return False

    def _generate_2d_fol_base(self, depth=0, rings=1):
        """Generates a 2D Flower of Life base structure."""
        node_id_counter = 0
        self._add_node(node_id_counter, (0,0,0), "primary", depth); node_id_counter+=1 # Center node
        
        for r in range(1, rings + 1):
            for i in range(6 * r):
                angle = (math.pi / (3*r)) * i
                x = self.radius * r * math.cos(angle)
                y = self.radius * r * math.sin(angle)
                self._add_node(node_id_counter, (x,y,0), "primary", depth); node_id_counter+=1
                if node_id_counter >= self.base_nodes_count: return


    def _generate_sphere_packing_base(self, num_nodes):
        """Generates base nodes using a simple sphere packing approximation (Fibonacci lattice)."""
        indices = np.arange(0, num_nodes, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / num_nodes)
        theta = np.pi * (1 + 5**0.5) * indices
        x = self.radius * np.cos(theta) * np.sin(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(phi)
        for i in range(num_nodes):
            self._add_node(i, (x[i], y[i], z[i]), "primary", 0)

    def _construct_layers(self, base_node_ids, max_depth):
        """ Recursively constructs higher-dimensional layers. """
        current_layer_nodes = base_node_ids
        all_higher_dim_nodes = []

        for d in range(1, max_depth + 1):
            new_nodes_this_depth = []
            for node_id in current_layer_nodes:
                base_coords = self.nodes[node_id]["coords"]
                # Create two new nodes "above" and "below" along a new dimension (e.g., w-axis for 4D)
                # The displacement uses self.radius, scaled by depth to maintain separation
                # For simplicity, new dimension is orthogonal.
                # A more complex model might use rotations or other transformations.
                
                # Create "positive" new dimension node
                new_node_id_pos = f"{node_id}_d{d}_pos" 
                # Simplified: extend into a new dimension by radius amount
                # For a true 3D to 4D etc., this needs more geometric rigor
                # Let's assume coords are (x,y,z) and we add a w-like component
                # For this example, we'll just use the node_id to ensure uniqueness
                # and place it "conceptually" in a higher dimension.
                # The coordinates will be tricky without defining the higher-D space.
                # Let's make a placeholder: new coords are base_coords + some offset in a new axis
                offset_vector = np.zeros(len(base_coords)) # Start with zeros
                # --- COMMENT REFINEMENT ---
                # The following line `np.append(base_coords, self.radius * d)` is a simplified placeholder
                # for generating coordinates in a higher dimension. True N-D geometric calculations
                # (e.g., using rotations or other transformations) would be required for a more accurate model.
                new_coords_pos = np.append(base_coords, self.radius * d) 
                
                if self._add_node(new_node_id_pos, new_coords_pos, "hyper", d, is_new_layer_node=True):
                    new_nodes_this_depth.append(new_node_id_pos)
                    self.adjacency[node_id].append(new_node_id_pos) # Connect base to new
                    self.adjacency[new_node_id_pos].append(node_id)

                # Create "negative" new dimension node
                new_node_id_neg = f"{node_id}_d{d}_neg"
                new_coords_neg = np.append(base_coords, -self.radius * d)

                if self._add_node(new_node_id_neg, new_coords_neg, "hyper", d, is_new_layer_node=True):
                    new_nodes_this_depth.append(new_node_id_neg)
                    self.adjacency[node_id].append(new_node_id_neg) # Connect base to new
                    self.adjacency[new_node_id_neg].append(node_id)
            
            if not new_nodes_this_depth: # Stop if no new nodes were added
                break
            
            # Compute adjacency for the newly created layer of hyper_nodes
            # This connects nodes within the same new depth level.
            self._compute_adjacency_for_layer(new_nodes_this_depth, num_neighbors=self.num_neighbors_setting)
            all_higher_dim_nodes.extend(new_nodes_this_depth)
            current_layer_nodes = new_nodes_this_depth # Next iteration builds upon these

    def _compute_adjacency_for_layer(self, node_ids_in_layer, num_neighbors):
        """Computes adjacency for nodes within a specific layer based on proximity."""
        if not node_ids_in_layer or len(node_ids_in_layer) < 2:
            return

        coords_map = {nid: self.nodes[nid]["coords"] for nid in node_ids_in_layer if nid in self.nodes}
        valid_node_ids = list(coords_map.keys())

        for i, node_id1 in enumerate(valid_node_ids):
            distances = []
            for j, node_id2 in enumerate(valid_node_ids):
                if i == j:
                    continue
                dist = np.linalg.norm(coords_map[node_id1] - coords_map[node_id2])
                distances.append((dist, node_id2))
            
            distances.sort(key=lambda x: x[0])
            
            for k in range(min(num_neighbors, len(distances))):
                neighbor_id = distances[k][1]
                if neighbor_id not in self.adjacency[node_id1]:
                    self.adjacency[node_id1].append(neighbor_id)
                if node_id1 not in self.adjacency[neighbor_id]: # Ensure bidirectionality
                    self.adjacency[neighbor_id].append(node_id1)

    def get_primary_nodes(self):
        """Returns nodes that are part of the base structure (depth 0 and not marked as new layer nodes)."""
        # This definition of primary might need adjustment based on how layers are built.
        # If base_nodes are those at depth 0, then filter by that.
        # Or, if "primary" means any node that isn't a "hyper" node from higher dimensions.
        return [self.nodes[nid] for nid in self.nodes if self.nodes[nid]["depth"] == 0 and not self.nodes[nid].get('is_new_layer_node', False)]

    def node_count(self):
        return len(self.nodes)

    def get_adjacency_list(self):
        return self.adjacency
    
    def get_node_info(self, node_id):
        return self.nodes.get(node_id)

# --- Core Bando Blocks ---
class BandoBlock:
    def __init__(self, dim):
        self.dim = dim
        self.W = np.random.randn(dim, dim) * 0.01 # Weight matrix
        self.b = np.zeros(dim) # Bias vector
        self.trainable = True

    def forward(self, x):
        # Basic linear transformation: y = xW + b
        return np.dot(x, self.W) + self.b

    def get_state_dict(self):
        return {"W": self.W, "b": self.b, "dim": self.dim, "class_name": self.__class__.__name__}

    def load_state_dict(self, state_dict):
        self.W = state_dict["W"]
        self.b = state_dict["b"]
        # self.dim is set by constructor. Only update if "dim" is explicitly in state_dict and different.
        # Or, more safely, ensure constructor always sets it, and here we only load W,b.
        # For Test 2, "dim" is intentionally removed from state_dict.
        # The orchestrator sets block_dim correctly during instantiation.
        # So, if "dim" is not in state_dict, we should rely on the already set self.dim.
        self.dim = state_dict.get("dim", self.dim)


    def summary(self):
        return f"{self.__class__.__name__}(dim={self.dim}, params={self.W.size + self.b.size})"

class VICtorchBlock(BandoBlock): # Stands for Vector-Input-Channel torch
    def __init__(self, dim, heads=4):
        super().__init__(dim)
        self.heads = heads
        assert dim % heads == 0, "Dimension must be divisible by number of heads."
        self.head_dim = dim // heads
        # Query, Key, Value weights for each head
        self.Wq = np.random.randn(heads, self.head_dim, self.head_dim) * 0.01
        self.Wk = np.random.randn(heads, self.head_dim, self.head_dim) * 0.01
        self.Wv = np.random.randn(heads, self.head_dim, self.head_dim) * 0.01
        self.Wo = np.random.randn(dim, dim) * 0.01 # Output projection

    def forward(self, x): # x is assumed to be (batch_size, dim) or just (dim,)
        if x.ndim == 1: x = x.reshape(1, -1) # Add batch dim if not present
        batch_size, _ = x.shape
        
        x_reshaped = x.reshape(batch_size, self.heads, self.head_dim) # (batch, heads, head_dim)
        
        q = np.einsum('bhd,hdo->bho', x_reshaped, self.Wq) # (batch, heads, head_dim)
        k = np.einsum('bhd,hdo->bho', x_reshaped, self.Wk)
        v = np.einsum('bhd,hdo->bho', x_reshaped, self.Wv)
        
        # Scaled dot-product attention per head
        # scores = np.einsum('bhd,bho->bho', q, k.transpose(0,2,1)) / np.sqrt(self.head_dim) # (batch, heads, heads) - This seems wrong, should be (batch, heads, sequence_len) if sequence
        scores = np.matmul(q, k.transpose(0,2,1)) / np.sqrt(self.head_dim) # q is (b,h,d), k.T is (b,d,h) -> result (b,h,h)
        
        # --- COMMENT REFINEMENT ---
        # NOTE: The attention mechanism here is significantly simplified due to the single vector input context.
        # Standard attention mechanisms operate over sequences of vectors. For a single input vector,
        # "self-attention" would typically imply interactions among its constituent parts (e.g., heads or sub-dimensions).
        # The current implementation uses a placeholder for `attention_weights` and directly passes `v` (value vectors)
        # as `attended_v`. This bypasses a meaningful attention calculation and serves as a structural placeholder.
        # A more developed implementation for single-vector attention might involve techniques like:
        # - Gating mechanisms.
        # - Different projection strategies for Q, K, V to enable relevant interactions.
        # - Component-wise attention if the "dimension" has sequence-like properties.
        attention_weights = np.random.rand(*scores.shape) # Placeholder for actual attention logic
        
        # Using V directly as a simplification, bypassing complex attention for a single vector input.
        attended_v = v # Simplified (batch, heads, head_dim)

        concatenated_output = attended_v.reshape(batch_size, self.dim) # (batch, dim)
        output = np.dot(concatenated_output, self.Wo) # (batch, dim)
        return output.squeeze() if batch_size == 1 else output


    def get_state_dict(self):
        base_state = super().get_state_dict()
        base_state.update({
            "heads": self.heads, "Wq": self.Wq, "Wk": self.Wk, "Wv": self.Wv, "Wo": self.Wo
        })
        return base_state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.heads = state_dict["heads"]
        self.head_dim = self.dim // self.heads
        self.Wq = state_dict["Wq"]
        self.Wk = state_dict["Wk"]
        self.Wv = state_dict["Wv"]
        self.Wo = state_dict["Wo"]

    def summary(self):
        total_params = self.W.size + self.b.size + self.Wq.size + self.Wk.size + self.Wv.size + self.Wo.size
        return f"{self.__class__.__name__}(dim={self.dim}, heads={self.heads}, params={total_params})"

class OmegaTensorBlock(BandoBlock): # High-dimensional tensor operations
    def __init__(self, dim, tensor_order=3):
        super().__init__(dim)
        self.tensor_order = tensor_order
        # Core tensor: (dim, dim, ..., dim) - order times
        self.core_tensor = np.random.randn(*([dim] * tensor_order)) * 0.01

    def forward(self, x): # x is (dim,)
        # Example: order 3, y_ijk = sum_a,b ( T_abk * x_i^a * x_j^b ) -> needs to map back to (dim,)
        # This is a complex operation to define generally.
        # Simplified: Contract x with the tensor in some way.
        # If order is 3 (d,d,d), x is (d,). Result should be (d,).
        # y_k = sum_ij (T_ijk * x_i * x_j) - still gives (d,)
        # This is computationally intensive.
        if self.tensor_order == 2: # Equivalent to standard BandoBlock matrix multiply
            return np.einsum('ij,j->i', self.core_tensor, x) if self.tensor_order == 2 else super().forward(x) # Fallback for order 2 for now
        elif self.tensor_order == 3:
            # y_k = sum_ij (T_ijk * x_i * x_j) -> This will be (dim,).
            # For simplicity, let's do something like: y_k = sum_i (T_iik * x_i)
            # This is just one way to contract. A more standard way might be mode-n product.
            # Let's try: y_k = sum_i,j (core_tensor_ijk * x_i * x_j) - this is still not right.
            # It should be y_c = sum_ab (T_abc * x_a * x_b)
             output = np.einsum('ijk,i,j->k', self.core_tensor, x, x) # Example for order 3
        else: # Fallback for other orders
            output = super().forward(x) # Or some other contraction
        return output


    def get_state_dict(self):
        base_state = super().get_state_dict()
        base_state.update({"tensor_order": self.tensor_order, "core_tensor": self.core_tensor})
        return base_state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.tensor_order = state_dict["tensor_order"]
        self.core_tensor = state_dict["core_tensor"]

    def summary(self):
        total_params = self.W.size + self.b.size + self.core_tensor.size
        return f"{self.__class__.__name__}(dim={self.dim}, order={self.tensor_order}, params={total_params})"


class FractalAttentionBlock(BandoBlock):
    def __init__(self, dim, depth=2, heads=2): # depth controls recursion
        super().__init__(dim)
        self.depth = depth
        self.heads = heads
        if dim > 0 and heads > 0 and dim % heads == 0 :
             self.sub_block_dim = dim // heads # Or some other division strategy
             # Create sub-blocks, which could be instances of VICtorchBlock or even FractalAttentionBlock
             self.sub_blocks = [VICtorchBlock(dim=self.sub_block_dim, heads=1) for _ in range(heads)] # Simplified
        else: # Handle cases where dim might be too small or zero
            self.sub_block_dim = 0
            self.sub_blocks = []


    def forward(self, x, current_depth=0): # x is (dim,)
        if current_depth >= self.depth or not self.sub_blocks or self.sub_block_dim == 0:
            return super().forward(x) # Base case: use standard BandoBlock linear transform

        # Split input x into parts for each sub_block / head
        # x is (dim,). Split into `self.heads` parts of size `self.sub_block_dim`.
        if x.ndim == 1:
            split_x = np.split(x, self.heads) if self.dim > 0 and self.heads > 0 and self.dim % self.heads == 0 else [x] # Handle non-divisible case simply
        else: # If x is batched (batch_size, dim)
            split_x = np.split(x, self.heads, axis=1) if self.dim > 0 and self.heads > 0 and self.dim % self.heads == 0 else [x]
        
        processed_parts = []
        for i, part_x in enumerate(split_x):
            if i < len(self.sub_blocks):
                 # Recursive call if sub-blocks are also FractalAttentionBlocks (not in this simple version)
                 # processed_parts.append(self.sub_blocks[i].forward(part_x, current_depth + 1))
                 processed_parts.append(self.sub_blocks[i].forward(part_x)) # Call VICtorchBlock
            else: # Should not happen if len(split_x) == len(self.sub_blocks)
                 processed_parts.append(part_x) 


        # Combine processed parts
        # If input was (dim,), output should be (dim,)
        # If input was (batch, dim), output should be (batch, dim)
        if not processed_parts: return x # Should not happen if x is valid

        if processed_parts[0].ndim == 1: # Each part is (sub_dim,)
            combined_output = np.concatenate(processed_parts) if len(processed_parts) > 0 else np.array([])
        else: # Each part is (batch, sub_dim)
            combined_output = np.concatenate(processed_parts, axis=1) if len(processed_parts) > 0 else np.array([[] for _ in range(x.shape[0])])


        # Final transform on combined output (optional, could be another BandoBlock)
        return super().forward(combined_output) if combined_output.size > 0 else combined_output


    def get_state_dict(self):
        base_state = super().get_state_dict()
        sub_block_states = [sb.get_state_dict() for sb in self.sub_blocks]
        base_state.update({"depth": self.depth, "heads": self.heads, "sub_block_dim": self.sub_block_dim, "sub_blocks": sub_block_states})
        return base_state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.depth = state_dict["depth"]
        self.heads = state_dict["heads"]
        self.sub_block_dim = state_dict.get("sub_block_dim", self.dim // self.heads if self.heads > 0 else self.dim) # Backward compat
        
        self.sub_blocks = []
        sub_block_states = state_dict.get("sub_blocks", [])
        for sb_state in sub_block_states:
            # Determine class of sub-block if stored, otherwise default (e.g. VICtorchBlock)
            # For this version, we assume sub_blocks are VICtorchBlock
            sb_class_name = sb_state.get("class_name", "VICtorchBlock") # Default if not specified
            # This is a simplification. A full system might need a class registry.
            if sb_class_name == "VICtorchBlock":
                block_dim = sb_state.get("dim", self.sub_block_dim)
                block_heads = sb_state.get("heads",1)
                sb = VICtorchBlock(dim=block_dim, heads=block_heads)
                sb.load_state_dict(sb_state)
                self.sub_blocks.append(sb)
            # Add elif for other sub-block types if necessary

    def summary(self):
        total_params = self.W.size + self.b.size
        for sb in self.sub_blocks: total_params += sum(p.size for p in sb.get_state_dict().values() if isinstance(p, np.ndarray))
        return f"{self.__class__.__name__}(dim={self.dim}, depth={self.depth}, heads={self.heads}, params ~{total_params})"

class MegaTransformerBlock(BandoBlock): # Conceptual: a very large transformer layer
    def __init__(self, dim, num_layers=6, heads=8, feedforward_dim_factor=4):
        super().__init__(dim)
        self.num_layers = num_layers
        self.heads = heads
        self.feedforward_dim = dim * feedforward_dim_factor
        # Represent layers as multiple VICtorchBlocks (for self-attention)
        # and BandoBlocks (for feedforward networks)
        self.attention_layers = [VICtorchBlock(dim, heads) for _ in range(num_layers)]
        self.feedforward_layers = [BandoBlock(dim) for _ in range(num_layers)] # Simplified FFN

    def forward(self, x): # x is (dim,) or (batch, dim)
        current_x = x
        for i in range(self.num_layers):
            # Self-attention layer (with residual connection and normalization - conceptual)
            attention_out = self.attention_layers[i].forward(current_x)
            # Add & Norm (simplified as just adding for now)
            current_x = current_x + attention_out # Residual connection
            
            # Feedforward layer (with residual connection and normalization - conceptual)
            ff_out = self.feedforward_layers[i].forward(current_x)
            # Add & Norm
            current_x = current_x + ff_out # Residual connection
        return current_x

    def get_state_dict(self):
        base_state = super().get_state_dict()
        attn_states = [l.get_state_dict() for l in self.attention_layers]
        ff_states = [l.get_state_dict() for l in self.feedforward_layers]
        base_state.update({
            "num_layers": self.num_layers, "heads": self.heads, 
            "feedforward_dim_factor": self.feedforward_dim // self.dim if self.dim > 0 else 4, # Store factor
            "attention_layers": attn_states, "feedforward_layers": ff_states
        })
        return base_state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.num_layers = state_dict["num_layers"]
        self.heads = state_dict["heads"]
        self.feedforward_dim = self.dim * state_dict["feedforward_dim_factor"]
        
        self.attention_layers = []
        for s in state_dict["attention_layers"]:
            l = VICtorchBlock(dim=s.get("dim", self.dim), heads=s.get("heads", self.heads))
            l.load_state_dict(s)
            self.attention_layers.append(l)
            
        self.feedforward_layers = []
        for s in state_dict["feedforward_layers"]:
            l = BandoBlock(dim=s.get("dim", self.dim)) # Assuming FFN layers are BandoBlocks
            l.load_state_dict(s)
            self.feedforward_layers.append(l)
    
    def summary(self):
        total_params = self.W.size + self.b.size # Base BandoBlock part (e.g. output projection)
        for l in self.attention_layers: total_params += sum(p.size for p in l.get_state_dict().values() if isinstance(p, np.ndarray))
        for l in self.feedforward_layers: total_params += sum(p.size for p in l.get_state_dict().values() if isinstance(p, np.ndarray))
        return f"{self.__class__.__name__}(dim={self.dim}, layers={self.num_layers}, heads={self.heads}, params ~{total_params})"


# --- Monolith combining blocks with a mesh ---
class BandoRealityMeshMonolith:
    def __init__(self, dim, mesh_depth=1, mesh_base_nodes=7, mesh_neighbors=3):
        self.dim = dim
        self.fm = FlowerOfLifeMesh3D(depth=mesh_depth, base_nodes=mesh_base_nodes, num_neighbors=mesh_neighbors)
        self.blocks = { # Pre-register some block types
            "BandoBlock": BandoBlock(dim),
            "VICtorchBlock": VICtorchBlock(dim),
            "OmegaTensorBlock": OmegaTensorBlock(dim),
            "FractalAttentionBlock": FractalAttentionBlock(dim),
            "MegaTransformerBlock": MegaTransformerBlock(dim)
        }
        # Can also dynamically add/replace blocks
        self.node_to_block_map = {} # node_id -> block_key
        self.output_aggregator = BandoBlock(dim) # To combine outputs

    def assign_block_to_node(self, node_id, block_key, block_params=None):
        if node_id not in self.fm.nodes:
            print(f"Warning: Node {node_id} not in mesh. Cannot assign block.")
            return
        if block_key not in self.blocks and block_params is not None : # Dynamically create if params given
             # This requires knowing the class from the key
             # Simplified: Assume block_key is a class name known globally or passed in
             try:
                 # --- COMMENT REFINEMENT ---
                 # Using `globals()[block_key]` to map a string to a class is a simplification
                 # suitable for this script's context. In more general or production systems,
                 # a dedicated registry pattern (e.g., a dictionary mapping names to classes)
                 # would be a more robust and safer way to manage and instantiate blocks.
                 block_class = globals()[block_key] 
                 self.blocks[block_key] = block_class(dim=self.dim, **block_params)
             except KeyError:
                 print(f"Error: Block class for key '{block_key}' not found.")
                 return
             except Exception as e:
                 print(f"Error instantiating block '{block_key}': {e}")
                 return

        elif block_key not in self.blocks:
            print(f"Warning: Block key {block_key} not registered and no params to create. Cannot assign.")
            return

        self.node_to_block_map[node_id] = block_key
        print(f"Assigned block {block_key} to node {node_id}")


    def mesh_forward(self, x_initial, node_sequence=None, k_iterations=3):
        # x_initial can be a single vector (dim,) or a dict {node_id: vector}
        # node_sequence: list of block_keys defining a path, or None for full mesh pass
        
        node_activations = {} # Store current activation for each node_id
        primary_nodes = self.fm.get_primary_nodes()
        if not primary_nodes: return x_initial # No mesh nodes to process

        # Initialize activations
        if isinstance(x_initial, dict):
            node_activations = x_initial.copy()
        else: # Single vector, apply to all primary nodes or a starting node
            # For simplicity, let's assume x_initial is for the first primary node if not a dict
            if primary_nodes:
                node_activations[primary_nodes[0]['id']] = x_initial


        if node_sequence: # Path traversal
            current_x = x_initial
            if not isinstance(x_initial, np.ndarray) or x_initial.shape != (self.dim,):
                 # If x_initial is not a single vector, try to get it from the first node in sequence (if mapped)
                 # This logic is a bit hand-wavy for path processing.
                 # Assume the sequence implies a conceptual data flow rather than strict mesh routing for now.
                 print("Warning: Path traversal expects a single initial vector. Using zero vector if needed.")
                 current_x = np.zeros(self.dim) if not isinstance(x_initial, np.ndarray) else x_initial


            for block_key in node_sequence:
                if block_key in self.blocks:
                    current_x = self.blocks[block_key].forward(current_x)
                else:
                    print(f"Warning: Block key {block_key} in sequence not found. Skipping.")
            return current_x # Output of the sequence

        # Full mesh pass (iterative updates)
        # Initialize all primary node activations if not already set
        for node_info in primary_nodes:
            nid = node_info['id']
            if nid not in node_activations:
                 node_activations[nid] = np.random.randn(self.dim) * 0.1 # Initialize with small random noise or zeros
                 # node_activations[nid] = np.zeros(self.dim)


        for iteration in range(k_iterations):
            print(f"Mesh iteration {iteration+1}")
            new_activations = {}
            for node_info in primary_nodes: # Iterate over primary nodes for processing
                node_id = node_info['id']
                
                # Aggregate inputs from neighbors
                neighbor_inputs_sum = np.zeros(self.dim)
                num_valid_neighbors = 0
                if node_id in self.fm.adjacency:
                    for neighbor_id in self.fm.adjacency[node_id]:
                        if neighbor_id in node_activations: # If neighbor has activation
                            neighbor_inputs_sum += node_activations[neighbor_id]
                            num_valid_neighbors += 1
                
                # Current node's own activation from previous step (or initial)
                prev_activation = node_activations.get(node_id, np.zeros(self.dim))
                
                # Effective input: combination of previous state and neighbor inputs
                # Simple averaging, could be more complex (e.g., weighted by edge properties)
                if num_valid_neighbors > 0:
                    effective_input = (prev_activation + neighbor_inputs_sum) / (1 + num_valid_neighbors)
                else:
                    effective_input = prev_activation

                # Process with the block assigned to this node
                block_key = self.node_to_block_map.get(node_id)
                if block_key and block_key in self.blocks:
                    output_activation = self.blocks[block_key].forward(effective_input)
                else: # Default behavior if no block or block not found: pass-through or dampen
                    output_activation = effective_input * 0.5 # Simple pass-through / attenuation
                
                new_activations[node_id] = output_activation
            node_activations = new_activations # Update all activations simultaneously for next iteration

        # Aggregate final outputs from all primary nodes
        final_output_sum = np.zeros(self.dim)
        num_contributing_nodes = 0
        for node_info in primary_nodes:
            nid = node_info['id']
            if nid in node_activations:
                final_output_sum += node_activations[nid]
                num_contributing_nodes +=1
        
        if num_contributing_nodes == 0: return np.zeros(self.dim) # Or handle error

        # Average or sum, then pass through final aggregator
        # final_aggregated_output = final_output_sum / len(primary_nodes) if primary_nodes else np.zeros(self.dim)
        final_aggregated_output = final_output_sum / num_contributing_nodes if num_contributing_nodes > 0 else np.zeros(self.dim)

        return self.output_aggregator.forward(final_aggregated_output)

    def get_state_dict(self):
        block_states = {key: block.get_state_dict() for key, block in self.blocks.items()}
        return {
            "dim": self.dim,
            "mesh_config": {"depth": self.fm.depth, "base_nodes": self.fm.base_nodes_count, "num_neighbors": self.fm.num_neighbors_setting},
            "blocks": block_states,
            "node_to_block_map": self.node_to_block_map,
            "output_aggregator": self.output_aggregator.get_state_dict()
        }

    def load_state_dict(self, state_dict):
        self.dim = state_dict["dim"]
        mesh_conf = state_dict["mesh_config"]
        self.fm = FlowerOfLifeMesh3D(depth=mesh_conf["depth"], base_nodes=mesh_conf["base_nodes"], num_neighbors=mesh_conf["num_neighbors"])
        
        self.blocks = {}
        for key, b_state in state_dict["blocks"].items():
            class_name = b_state.get("class_name", key) # Use key as fallback for older saves
            # Need a robust way to get class from class_name string
            try:
                BlockClass = globals()[class_name] # Assumes classes are in global scope
                block_instance = BlockClass(dim=b_state.get("dim", self.dim)) # Pass dim if available in state
                block_instance.load_state_dict(b_state)
                self.blocks[key] = block_instance
            except KeyError:
                print(f"Error: Block class '{class_name}' (key: {key}) not found during load. Skipping.")
            except Exception as e:
                print(f"Error loading block '{key}': {e}")


        self.node_to_block_map = state_dict["node_to_block_map"]
        self.output_aggregator = BandoBlock(self.dim) # Create new instance
        self.output_aggregator.load_state_dict(state_dict["output_aggregator"])

    def summary(self):
        s = f"BandoRealityMeshMonolith(dim={self.dim}, mesh_nodes={self.fm.node_count()})\n"
        s += "Registered Blocks:\n"
        for key, block in self.blocks.items():
            s += f"  - {key}: {block.summary()}\n"
        s += "Node Assignments:\n"
        for nid, bkey in self.node_to_block_map.items():
            s += f"  - Node {nid} -> {bkey}\n"
        s += f"Output Aggregator: {self.output_aggregator.summary()}"
        return s


# --- Router and Coordinator ---
class MeshRouter:
    def __init__(self, flower_of_life_mesh, node_models, k_iterations=3, attenuation=0.5):
        self.mesh = flower_of_life_mesh
        self.node_models = node_models # List of BandoBlock instances, aligned with primary node indices
        self.k_iterations = k_iterations
        self.attenuation = attenuation # Factor for how much neighbor influence decays
        self.primary_node_ids = [pn['id'] for pn in self.mesh.get_primary_nodes()]
        if len(self.node_models) != len(self.primary_node_ids):
            print(f"Warning: Number of node models ({len(self.node_models)}) does not match number of primary mesh nodes ({len(self.primary_node_ids)}). Router may behave unexpectedly.")


    def process(self, initial_activations): # initial_activations: list or dict
        """
        Processes activations through the mesh.
        initial_activations: A list of initial activation vectors (np.array) for each primary node,
                             or a dictionary {node_id: activation_vector}.
        """
        if not self.primary_node_ids: return []

        # Determine a default dimension for activations if not determinable from a specific model
        default_dim_router = 0
        if self.node_models:
            first_valid_model = next((m for m in self.node_models if m is not None), None)
            if first_valid_model:
                default_dim_router = first_valid_model.dim
        
        if default_dim_router == 0 and isinstance(initial_activations, list) and initial_activations:
            first_valid_activation = next((act for act in initial_activations if act is not None and hasattr(act, 'shape') and act.ndim > 0 and act.shape[0]>0), None)
            if first_valid_activation:
                default_dim_router = first_valid_activation.shape[0]
        elif default_dim_router == 0 and isinstance(initial_activations, dict) and initial_activations:
             first_valid_activation = next((act for act in initial_activations.values() if act is not None and hasattr(act, 'shape') and act.ndim > 0 and act.shape[0]>0), None)
             if first_valid_activation:
                default_dim_router = first_valid_activation.shape[0]

        if default_dim_router == 0: # Still zero, this is a fallback
            # This might happen if node_models is empty or all None, and initial_activations are also all None or empty.
            # Try to get it from mesh's model_dim if possible, but router doesn't know it directly.
            # As a last resort, use a placeholder or raise error. For now, print warning and use 1.
            # Standardized Warning Message
            print("Warning: MeshRouter could not determine a consistent default dimension. Using fallback dimension 1. This may lead to errors if not intended.")
            default_dim_router = 1

        current_activations = {}
        if isinstance(initial_activations, list):
            if len(initial_activations) != len(self.primary_node_ids):
                print(f"Error: Length of initial_activations list ({len(initial_activations)}) must match number of primary nodes ({len(self.primary_node_ids)}).")
                # Initialize with default_dim_router to prevent (0,) shapes if list is too short and models are None
                for i, nid in enumerate(self.primary_node_ids):
                    current_activations[nid] = initial_activations[i] if i < len(initial_activations) and initial_activations[i] is not None else \
                                               np.zeros(self.node_models[i].dim if i < len(self.node_models) and self.node_models[i] else default_dim_router)
            else: # Correct length list
                 for i, nid in enumerate(self.primary_node_ids):
                    current_activations[nid] = initial_activations[i] if initial_activations[i] is not None else \
                                               np.zeros(self.node_models[i].dim if i < len(self.node_models) and self.node_models[i] else default_dim_router)
        elif isinstance(initial_activations, dict):
            current_activations = initial_activations.copy() # Assume dict provides valid shapes or None
            # Ensure all primary nodes get an entry, even if not in the dict
            for i, nid in enumerate(self.primary_node_ids):
                if nid not in current_activations:
                    current_activations[nid] = np.zeros(self.node_models[i].dim if i < len(self.node_models) and self.node_models[i] else default_dim_router)
                elif current_activations[nid] is None: # If dict provided a None value
                    current_activations[nid] = np.zeros(self.node_models[i].dim if i < len(self.node_models) and self.node_models[i] else default_dim_router)

        else: # Single vector applied to all, or error (this path might need review for default_dim_router usage)
            print("Error: initial_activations should be a list or dict.") # This case is problematic.
            # If it's a single vector, it should have been handled by orchestrator to make a list.
            # Returning list of zeros based on model dims or default_dim_router
            return [np.zeros(model.dim if model else default_dim_router) for model in self.node_models]


        # Ensure all primary nodes in current_activations have a valid np.array (e.g. if dict had None)
        # and correct dimension if possible.
        for i, nid in enumerate(self.primary_node_ids):
            node_model_dim = self.node_models[i].dim if i < len(self.node_models) and self.node_models[i] else default_dim_router
            if nid not in current_activations or current_activations[nid] is None:
                current_activations[nid] = np.zeros(node_model_dim)
            elif not isinstance(current_activations[nid], np.ndarray) or current_activations[nid].shape[0] != node_model_dim:
                # This handles cases where a dict might provide incorrectly shaped arrays.
                # Forcing to default_dim_router or node_model_dim.
                # print(f"Warning: Activation for node {nid} has incorrect shape {current_activations[nid].shape if hasattr(current_activations[nid], 'shape') else 'N/A'}. Resetting to zeros({node_model_dim}).")
                current_activations[nid] = np.zeros(node_model_dim)


        for iteration in range(self.k_iterations):
            next_activations = {}
            for idx, node_id in enumerate(self.primary_node_ids):
                node_model = self.node_models[idx] if idx < len(self.node_models) else None
                if node_model is None: # Skip if no model for this node
                    # Carry over activation or set to zero
                    next_activations[node_id] = current_activations.get(node_id, np.zeros(1)) # Problem if dim unknown
                    continue

                # Gather activations from neighbors
                neighbor_sum = np.zeros(node_model.dim)
                num_neighbors = 0
                if node_id in self.mesh.adjacency:
                    for neighbor_id in self.mesh.adjacency[node_id]:
                        if neighbor_id in current_activations: # Consider only primary nodes for now
                            neighbor_sum += current_activations[neighbor_id] * self.attenuation
                            num_neighbors += 1
                
                # Combine with current node's activation
                # Input to the model is a mix of its current state and influenced neighbor states
                # This is a simple model; could be more sophisticated (e.g. weighted by distance)
                input_for_model = current_activations.get(node_id, np.zeros(node_model.dim)) + neighbor_sum
                if num_neighbors > 0 : input_for_model /= (1+num_neighbors*self.attenuation) # Normalize influence somewhat


                next_activations[node_id] = node_model.forward(input_for_model)
            current_activations = next_activations
        
        # Return activations in the order of primary_node_ids
        return [current_activations.get(nid) for nid in self.primary_node_ids]


class HeadCoordinatorBlock(BandoBlock):
    def __init__(self, dim, hidden_dim, output_dim): # dim is total input dim from all FOL nodes
        super().__init__(dim) # Input W,b are not directly used like this from BandoBlock
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Override W,b from BandoBlock for specific coordinator layers
        self.W1 = np.random.randn(dim, hidden_dim) * 0.01 # Input to Hidden
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01 # Hidden to Output
        self.b2 = np.zeros(output_dim)

    def forward(self, aggregated_fol_output): # aggregated_fol_output is a flat vector
        # aggregated_fol_output shape should be (dim,)
        if aggregated_fol_output.shape[0] != self.W1.shape[0]:
            # Try to pad or truncate if there's a mismatch. This can happen if num_nodes or model_dim changes.
            # This is a simplistic fix. A robust solution might need architectural changes or error handling.
            print(f"Warning: HeadCoordinator input dim mismatch. Expected {self.W1.shape[0]}, got {aggregated_fol_output.shape[0]}. Adjusting...")
            target_dim = self.W1.shape[0]
            current_dim = aggregated_fol_output.shape[0]
            if current_dim < target_dim: # Pad with zeros
                padding = np.zeros(target_dim - current_dim)
                aggregated_fol_output = np.concatenate((aggregated_fol_output, padding))
            else: # Truncate
                aggregated_fol_output = aggregated_fol_output[:target_dim]


        h = np.dot(aggregated_fol_output, self.W1) + self.b1
        h_activated = np.tanh(h) # Example activation: tanh
        output = np.dot(h_activated, self.W2) + self.b2
        return output

    def get_state_dict(self):
        # Don't call super().get_state_dict() as W,b are different here
        return {
            "dim": self.W1.shape[0], # Input dim to W1
            "hidden_dim": self.hidden_dim, 
            "output_dim": self.output_dim,
            "W1": self.W1, "b1": self.b1, 
            "W2": self.W2, "b2": self.b2,
            "class_name": self.__class__.__name__
        }

    def load_state_dict(self, state_dict):
        # self.dim = state_dict["input_dim"] # Keep this to match BandoBlock parent if needed for other things
        self.hidden_dim = state_dict["hidden_dim"]
        self.output_dim = state_dict["output_dim"]
        self.W1 = state_dict["W1"]
        self.b1 = state_dict["b1"]
        self.W2 = state_dict["W2"]
        self.b2 = state_dict["b2"]
        # Also update self.dim from BandoBlock if it's meant to represent the input dim for W1
        self.dim = self.W1.shape[0]


# --- Orchestrator ---
class FlowerOfLifeNetworkOrchestrator:
    def __init__(self, num_nodes, model_dim, 
                 mesh_depth=1, mesh_base_nodes=None, mesh_num_neighbors=6, 
                 k_ripple_iterations=3, router_attenuation=0.5,
                 coordinator_hidden_dim=128, coordinator_output_dim=None):
        
        self.num_nodes = num_nodes # Number of primary nodes in the FoL mesh
        self.model_dim = model_dim # Dimension of model at each node
        
        if mesh_base_nodes is None: mesh_base_nodes = num_nodes # Default base_nodes to num_nodes

        self.mesh = FlowerOfLifeMesh3D(depth=mesh_depth, base_nodes=mesh_base_nodes, 
                                       compute_adjacency_for_base=True, num_neighbors=mesh_num_neighbors)
        
        # Ensure num_nodes matches actual primary nodes generated if different from mesh_base_nodes
        # This can happen if mesh_base_nodes implies a structure (e.g. 7 for FoL) but user requests different num_nodes
        # For now, we assume num_nodes will be respected by MeshRouter by aligning models list.
        # If mesh generates N primary nodes, and self.num_nodes = M, router will use M models.
        # This might lead to mismatch if M != N.
        # A safer way: self.num_nodes = len(self.mesh.get_primary_nodes()) if mesh_base_nodes was used to define structure.
        # Let's assume for now that mesh_base_nodes and num_nodes are consistent or handled by router.
        # If mesh_base_nodes was set to define a specific structure (e.g. 7 for FoL base),
        # then the actual number of primary nodes might be fixed by that structure.
        # Let's use the count from the generated mesh's primary nodes as the definitive num_nodes.
        actual_primary_nodes = len(self.mesh.get_primary_nodes())
        if actual_primary_nodes != self.num_nodes:
            # Standardized Warning Message
            print(f"Warning: Requested num_nodes ({self.num_nodes}) differs from mesh's actual primary nodes ({actual_primary_nodes}). Using actual count: {actual_primary_nodes}.")
            self.num_nodes = actual_primary_nodes


        self.node_models = [None] * self.num_nodes # Stores BandoBlock instances
        self.available_block_classes = { # Registry of known block types
            "BandoBlock": BandoBlock,
            "VICtorchBlock": VICtorchBlock,
            "OmegaTensorBlock": OmegaTensorBlock,
            "FractalAttentionBlock": FractalAttentionBlock,
            "MegaTransformerBlock": MegaTransformerBlock
        }

        self.router = MeshRouter(self.mesh, self.node_models, # node_models passed by reference, updated by assign_block
                                 k_iterations=k_ripple_iterations, attenuation=router_attenuation)
        
        coordinator_input_dim = self.num_nodes * self.model_dim # Aggregated output from all nodes
        if coordinator_output_dim is None: coordinator_output_dim = model_dim # Default to model_dim
        self.head_coordinator = HeadCoordinatorBlock(dim=coordinator_input_dim, 
                                                     hidden_dim=coordinator_hidden_dim, 
                                                     output_dim=coordinator_output_dim)

    def assign_block_to_node(self, node_index, block_class_name, **block_params):
        if not (0 <= node_index < self.num_nodes):
            print(f"Error: Node index {node_index} is out of range (0-{self.num_nodes-1}).")
            return

        if block_class_name not in self.available_block_classes:
            print(f"Error: Block class '{block_class_name}' not recognized.")
            return
        
        BlockClass = self.available_block_classes[block_class_name]
        # Ensure 'dim' is passed if not explicitly in block_params, using self.model_dim
        if 'dim' not in block_params:
            block_params['dim'] = self.model_dim
        
        try:
            instance = BlockClass(**block_params)
            self.node_models[node_index] = instance
            # Update router's view of models (since it holds a reference, this should be automatic)
            # self.router.node_models = self.node_models # Re-assign if it was a copy
            print(f"Assigned {block_class_name} to node {node_index} (ID: {self.router.primary_node_ids[node_index] if node_index < len(self.router.primary_node_ids) else 'N/A'}).")
        except Exception as e:
            print(f"Error instantiating block {block_class_name}: {e}")


    def process_input(self, network_input):
        """
        Processes input through the FOL network.
        network_input: Can be a single vector (np.array of shape (model_dim,)) to be broadcast
                       to all nodes, or a list of vectors (each for a node),
                       or a dictionary {node_id: vector}.
        """
        if not self.node_models or all(m is None for m in self.node_models):
             print("Warning: No models assigned to nodes. Network cannot process input meaningfully.")
             # Depending on desired behavior, could return zeros, None, or raise error.
             return np.zeros(self.head_coordinator.output_dim if self.head_coordinator else self.model_dim)


        initial_activations_list = [None] * self.num_nodes

        if isinstance(network_input, np.ndarray) and network_input.shape == (self.model_dim,):
            # Single vector, broadcast to all nodes that have a model
            for i in range(self.num_nodes):
                if self.node_models[i] is not None:
                    initial_activations_list[i] = network_input.copy()
                else: # Node has no model, initialize with zeros or handle as per router
                    initial_activations_list[i] = np.zeros(self.model_dim)
        elif isinstance(network_input, list):
            if len(network_input) == self.num_nodes:
                for i in range(self.num_nodes):
                    if network_input[i] is not None and network_input[i].shape == (self.model_dim,):
                         initial_activations_list[i] = network_input[i]
                    elif self.node_models[i] is not None : # Input is None or wrong shape, but model exists
                         initial_activations_list[i] = np.zeros(self.model_dim) # Default to zeros
                    # If network_input[i] is None and self.node_models[i] is None, it remains None (handled by router)
            else:
                print(f"Error: Input list length ({len(network_input)}) must match num_nodes ({self.num_nodes}).")
                return None # Or raise error
        elif isinstance(network_input, dict): # Dict {node_id: vector} - convert to list for router
            # This requires mapping node_ids to indices if router expects a list.
            # Assuming router's primary_node_ids gives the order.
            temp_activations_map = network_input 
            initial_activations_list = [np.zeros(self.model_dim)] * self.num_nodes # Default to zeros
            for i, nid in enumerate(self.router.primary_node_ids):
                if i < self.num_nodes : # Ensure we don't go out of bounds for initial_activations_list
                    if nid in temp_activations_map and temp_activations_map[nid] is not None and temp_activations_map[nid].shape == (self.model_dim,):
                        initial_activations_list[i] = temp_activations_map[nid]
                    # else it remains zeros (or whatever default was set)
        else:
            print("Error: Invalid network_input format.")
            return None # Or raise error

        # Router processes the list of activations
        # The router itself should handle None entries in initial_activations_list (e.g. by using zeros)
        routed_outputs = self.router.process(initial_activations_list)
        
        # Aggregate outputs from router for HeadCoordinator
        # routed_outputs is a list of vectors, one for each primary node
        # Filter out None results if any node model failed or was absent
        valid_outputs = [out for out in routed_outputs if out is not None]
        if not valid_outputs:
            print("Warning: Router produced no valid outputs. HeadCoordinator cannot process.")
            return np.zeros(self.head_coordinator.output_dim if self.head_coordinator else self.model_dim)

        # Concatenate all node outputs into a single flat vector
        # Ensure all outputs have the expected dimension; pad/truncate if necessary.
        # This can be complex if dimensions vary unexpectedly. For now, assume they match self.model_dim.
        processed_outputs = []
        for out_vec in valid_outputs:
            if out_vec.shape[0] == self.model_dim:
                processed_outputs.append(out_vec)
            elif out_vec.shape[0] < self.model_dim: # Pad
                padding = np.zeros(self.model_dim - out_vec.shape[0])
                processed_outputs.append(np.concatenate((out_vec, padding)))
            else: # Truncate
                processed_outputs.append(out_vec[:self.model_dim])
        
        # If some nodes didn't output (e.g. no model), fill with zeros for those spots before concat
        # to maintain fixed input size for coordinator.
        # The router should return a list of length self.num_nodes, with zeros for missing models.
        # So, len(routed_outputs) should be self.num_nodes.
        if len(routed_outputs) != self.num_nodes:
            # This case should ideally be handled by the router ensuring output list matches num_nodes
            # Standardized Warning Message
            print(f"Warning: Router output length ({len(routed_outputs)}) mismatches num_nodes ({self.num_nodes}). Padding coordinator input with zeros.")
            # Create a full list of zeros and fill in what we have
            full_outputs_for_concat = [np.zeros(self.model_dim) for _ in range(self.num_nodes)]
            for i, out_vec in enumerate(routed_outputs): # Assuming routed_outputs corresponds to first N nodes if shorter
                if i < self.num_nodes and out_vec is not None:
                     # Ensure correct dimension before assignment
                     if out_vec.shape[0] == self.model_dim: full_outputs_for_concat[i] = out_vec
                     elif out_vec.shape[0] < self.model_dim: full_outputs_for_concat[i] = np.concatenate((out_vec, np.zeros(self.model_dim - out_vec.shape[0])))
                     else: full_outputs_for_concat[i] = out_vec[:self.model_dim]

            aggregated_input_for_coordinator = np.concatenate(full_outputs_for_concat) if full_outputs_for_concat else np.zeros(self.num_nodes * self.model_dim)

        else: # Correct number of outputs from router
            # Ensure all elements are arrays of correct dimension before concatenation
            final_concat_list = []
            for i in range(self.num_nodes):
                vec = routed_outputs[i]
                if vec is None: vec = np.zeros(self.model_dim) # Replace None with zeros
                elif vec.shape[0] != self.model_dim: # Adjust dimension if needed
                    if vec.shape[0] < self.model_dim: vec = np.concatenate((vec, np.zeros(self.model_dim - vec.shape[0])))
                    else: vec = vec[:self.model_dim]
                final_concat_list.append(vec)
            aggregated_input_for_coordinator = np.concatenate(final_concat_list) if final_concat_list else np.zeros(self.num_nodes * self.model_dim)


        if aggregated_input_for_coordinator.shape[0] != self.head_coordinator.W1.shape[0]:
             # This check is also inside HeadCoordinator, but good to be aware here
             print(f"Warning: Aggregated input dim {aggregated_input_for_coordinator.shape[0]} " \
                   f"mismatch for HeadCoordinator (expected {self.head_coordinator.W1.shape[0]}).")
             # HeadCoordinator itself has logic to pad/truncate, so we can pass it as is.

        final_response = self.head_coordinator.forward(aggregated_input_for_coordinator)
        return final_response

    def save_network_state(self, file_path: str) -> bool:
        try:
            node_model_states = []
            for model in self.node_models:
                if model:
                    node_model_states.append({
                        "class_name": model.__class__.__name__,
                        "state_dict": model.get_state_dict()
                    })
                else:
                    node_model_states.append(None)
            
            network_state = {
                "num_nodes": self.num_nodes,
                "model_dim": self.model_dim,
                "mesh_config": { 
                    "depth": self.mesh.depth,
                    "radius": self.mesh.radius,
                    "base_nodes": self.mesh.base_nodes_count,
                    "compute_adjacency_for_base": True, # Assuming it was true if mesh exists
                    "num_neighbors": self.mesh.num_neighbors_setting # Use the setting used for creation
                },
                "router_config": {
                    "k_iterations": self.router.k_iterations,
                    "attenuation": self.router.attenuation
                },
                "node_model_states": node_model_states,
                "head_coordinator_state": self.head_coordinator.get_state_dict()
            }
            with open(file_path, "wb") as f:
                pickle.dump(network_state, f)
            print(f"FlowerOfLifeNetworkOrchestrator state saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving network state: {e}")
            return False

    def load_network_state(self, file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                network_state = pickle.load(f)

            self.model_dim = network_state["model_dim"] # Load model_dim first
            # self.num_nodes = network_state["num_nodes"] # num_nodes will be determined by mesh config or re-set

            
            mesh_conf = network_state.get("mesh_config", {
                "depth": 1, "radius": 1.0, 
                "base_nodes": network_state["num_nodes"], # Use loaded num_nodes for base_nodes if no specific config
                "compute_adjacency_for_base": True, 
                "num_neighbors": 6 
            })
            # If 'base_nodes' from loaded state is different from network_state["num_nodes"],
            # it implies the mesh structure itself defines the number of primary nodes.
            self.mesh = FlowerOfLifeMesh3D(
                depth=mesh_conf["depth"], radius=mesh_conf["radius"], base_nodes=mesh_conf["base_nodes"],
                compute_adjacency_for_base=mesh_conf.get("compute_adjacency_for_base", True), 
                num_neighbors=mesh_conf["num_neighbors"]
            )
            # Update num_nodes based on the loaded mesh's actual primary node count
            self.num_nodes = len(self.mesh.get_primary_nodes())
            print(f"Loaded mesh resulted in {self.num_nodes} primary nodes.")


            self.node_models = [None] * self.num_nodes # Initialize with correct number of Nones
            loaded_node_model_states = network_state["node_model_states"]
            
            # Adjust loaded_node_model_states list length if it mismatches new self.num_nodes
            if len(loaded_node_model_states) != self.num_nodes:
                print(f"Warning: Saved node_model_states count ({len(loaded_node_model_states)}) "
                      f"differs from new mesh's primary node count ({self.num_nodes}). Adjusting list.")
                # Pad with Nones if new mesh has more nodes
                while len(loaded_node_model_states) < self.num_nodes:
                    loaded_node_model_states.append(None)
                # Truncate if new mesh has fewer nodes
                loaded_node_model_states = loaded_node_model_states[:self.num_nodes]


            for i, model_state_info in enumerate(loaded_node_model_states):
                if i >= self.num_nodes: break # Should be handled by list adjustment above, but as safeguard
                if model_state_info:
                    class_name = model_state_info["class_name"]
                    state_dict = model_state_info["state_dict"]
                    block_class = self.available_block_classes.get(class_name)
                    if block_class:
                        # Use block's own dim if saved, else current orchestrator's model_dim
                        block_dim = state_dict.get("dim", self.model_dim) 
                        try:
                            # Pass all params from state_dict that are constructor args (excluding 'dim' handled above)
                            # This is tricky; for now, assume 'dim' is the main one, others are specific like 'heads'
                            # A better way is for blocks to have a `from_state_dict` class method or more structured params.
                            # Simplification: pass only dim, specific blocks handle their params from state_dict.
                            # Constructor params often include more than just 'dim'.
                            # E.g. VICtorchBlock needs 'heads'. Fractal needs 'depth', 'heads'.
                            # Let's try to pass relevant params from the state_dict if they exist as keys.
                            # --- COMMENT REFINEMENT ---
                            # The following extraction of constructor parameters (e.g., 'heads', 'depth')
                            # directly from the state_dict for block instantiation is an ad-hoc simplification
                            # specific to this script. A more robust and maintainable approach would involve:
                            #   1. Blocks defining a `from_config` or `from_state_dict` class method that
                            #      knows how to extract its necessary parameters.
                            #   2. A clearer schema or specification for what each block's state_dict should contain
                            #      regarding constructor arguments vs. loadable weights/attributes.
                            constructor_params = {'dim': block_dim}
                            if 'heads' in state_dict and (class_name == "VICtorchBlock" or class_name == "FractalAttentionBlock" or class_name == "MegaTransformerBlock"):
                                constructor_params['heads'] = state_dict['heads']
                            if 'depth' in state_dict and class_name == "FractalAttentionBlock":
                                constructor_params['depth'] = state_dict['depth']
                            if 'num_layers' in state_dict and class_name == "MegaTransformerBlock":
                                 constructor_params['num_layers'] = state_dict['num_layers']
                            if 'feedforward_dim_factor' in state_dict and class_name == "MegaTransformerBlock":
                                 constructor_params['feedforward_dim_factor'] = state_dict['feedforward_dim_factor']
                            if 'tensor_order' in state_dict and class_name == "OmegaTensorBlock":
                                 constructor_params['tensor_order'] = state_dict['tensor_order']


                            instance = block_class(**constructor_params)
                            instance.load_state_dict(state_dict)
                            self.node_models[i] = instance
                        except Exception as e_inst:
                             print(f"Error instantiating/loading state for block {class_name} at node {i}: {e_inst}")
                             import traceback
                             traceback.print_exc() # Keep traceback for this critical error
                    else:
                        # Standardized Warning Message
                        print(f"Warning: Block class '{class_name}' for node {i} not found in available_block_classes. Node model will be None.")
            
            router_conf = network_state.get("router_config", {"k_iterations":3, "attenuation":0.5})
            self.router = MeshRouter(self.mesh, self.node_models, 
                                     k_iterations=router_conf["k_iterations"], 
                                     attenuation=router_conf["attenuation"])
            
            head_coord_state = network_state["head_coordinator_state"]
            # Coordinator's input dim should be recalced based on current num_nodes * model_dim
            coord_input_dim = self.num_nodes * self.model_dim
            # Use saved hidden/output dims, but input dim must match current network structure
            coord_hidden_dim = head_coord_state.get("hidden_dim", 128) 
            coord_output_dim = head_coord_state.get("output_dim", self.model_dim)


            self.head_coordinator = HeadCoordinatorBlock(dim=coord_input_dim, 
                                                         hidden_dim=coord_hidden_dim, 
                                                         output_dim=coord_output_dim)
            # The loaded state for HeadCoordinator might have W1 with different input dim.
            # HeadCoordinator's load_state_dict needs to be robust or we need to re-init W1 if dims changed.
            # For now, assume HeadCoordinator.load_state_dict handles this (e.g. by using the new dim for W1 if shapes mismatch)
            # Or, more simply, the loaded state's W1.shape[0] will define its input dim.
            # Let's ensure the coordinator is created with the *loaded* input dim for W1 if that's intended.
            # The current HeadCoordinator.load_state_dict updates self.dim from W1.shape[0].
            # So, create with potentially new coord_input_dim, then load_state_dict will adjust its internal self.dim.
            self.head_coordinator.load_state_dict(head_coord_state)
            
            print(f"FlowerOfLifeNetworkOrchestrator state loaded from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading network state: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    np.random.seed(777); dim_ex=32; x_in=np.random.randn(dim_ex) # Changed from (dim_ex, dim_ex) to (dim_ex,) for single vector tests
    print("\n--- Testing FlowerOfLifeMesh3D ---")
    fol_tst=FlowerOfLifeMesh3D(depth=1,radius=1.0,base_nodes=7,compute_adjacency_for_base=True,num_neighbors=3)
    print(f"FOLMesh3D (7 nodes, depth 1) node count: {fol_tst.node_count()}") # Will be > 7 due to depth
    p_nodes=fol_tst.get_primary_nodes(); print(f"Primary nodes: {len(p_nodes)}") # Should be 7
    if p_nodes: print(f"Adj for node 0 ('{p_nodes[0]['id']}') in primary layer: {fol_tst.adjacency.get(p_nodes[0]['id'])}")
    
    # Test a hyper node if exists
    hyper_nodes_exist = any(ninfo['type'] == 'hyper' for nid, ninfo in fol_tst.nodes.items())
    if hyper_nodes_exist:
        first_hyper_node = next(nid for nid, ninfo in fol_tst.nodes.items() if ninfo['type'] == 'hyper')
        print(f"Adj for a hyper node '{first_hyper_node}': {fol_tst.adjacency.get(first_hyper_node)}")


    print("\n--- Testing BandoRealityMeshMonolith ---")
    # Monolith test requires single vector input if node_sequence is used, or dict for general mesh_forward
    mono_dim = 16 # Use a smaller dim for monolith to speed up if needed
    mono_x_in = np.random.randn(mono_dim)
    mono=BandoRealityMeshMonolith(dim=mono_dim, mesh_depth=0, mesh_base_nodes=3, mesh_neighbors=2) # Simpler mesh for monolith test
    print(f">>> Monolith internal mesh node count: {mono.fm.node_count()} (Primary: {len(mono.fm.get_primary_nodes())})")
    
    # Assign some blocks to nodes for monolith test
    primary_nodes_mono = mono.fm.get_primary_nodes()
    if len(primary_nodes_mono) >= 1: mono.assign_block_to_node(primary_nodes_mono[0]['id'], "VICtorchBlock")
    if len(primary_nodes_mono) >= 2: mono.assign_block_to_node(primary_nodes_mono[1]['id'], "FractalAttentionBlock")
    if len(primary_nodes_mono) >= 3: mono.assign_block_to_node(primary_nodes_mono[2]['id'], "BandoBlock")

    # Test mesh_forward with full mesh pass (iterative)
    print("Testing monolith mesh_forward (full pass)...")
    out_mf_full = mono.mesh_forward(x_initial=mono_x_in, k_iterations=2) # x_initial applied to first primary node
    print(f">>> Output shape after full mesh_forward: {out_mf_full.shape}")
    
    # Test mesh_forward with node_sequence
    print("Testing monolith mesh_forward (sequence)...")
    out_mf_seq = mono.mesh_forward(x_initial=mono_x_in, node_sequence=["VICtorchBlock","FractalAttentionBlock","MegaTransformerBlock"])
    print(f">>> Output shape after mesh_forward (sequence): {out_mf_seq.shape}")
    print(f">>> Monolith summary: {mono.summary()}")


    print("\n--- Testing Block Save/Load ---")
    vt_b=VICtorchBlock(dim=dim_ex); vt_b.Wq[0,0]=123.456; sd_vt=vt_b.get_state_dict()
    n_vt_b=VICtorchBlock(dim=dim_ex); n_vt_b.load_state_dict(sd_vt); assert (n_vt_b.Wq[0,0]==123.456).all(), "VTBlock load fail"
    print("VICtorchBlock save/load test PASSED.")

    print("\n--- Testing Monolith Save/Load ---")
    # Modify a block within the monolith for testing save/load
    # Ensure block exists, e.g. the one assigned to the first primary node or a default one
    target_block_key_mono_save_test = None
    if primary_nodes_mono and mono.node_to_block_map.get(primary_nodes_mono[0]['id']):
        target_block_key_mono_save_test = mono.node_to_block_map[primary_nodes_mono[0]['id']]
    elif "VICtorchBlock" in mono.blocks: # Fallback to a registered block if no assignment
         target_block_key_mono_save_test = "VICtorchBlock"

    if target_block_key_mono_save_test and hasattr(mono.blocks[target_block_key_mono_save_test], 'Wq'):
        mono.blocks[target_block_key_mono_save_test].Wq[0,1]=789.123
        print(f"Modified {target_block_key_mono_save_test} for save/load test.")
    else:
        print(f"Could not find suitable block (VICtorchBlock with Wq) in monolith to modify for save/load test. Test may be less effective.")

    sd_m=mono.get_state_dict()
    with open("temp_monolith_test.pkl","wb") as f_pkl: pickle.dump(sd_m,f_pkl) 
    with open("temp_monolith_test.pkl","rb") as f_pkl_rb: lsd_m=pickle.load(f_pkl_rb) 
    
    n_mono=BandoRealityMeshMonolith(dim=mono_dim, mesh_depth=0, mesh_base_nodes=3) # Create new instance with compatible params
    n_mono.load_state_dict(lsd_m)
    
    if target_block_key_mono_save_test and hasattr(n_mono.blocks.get(target_block_key_mono_save_test), 'Wq'):
        assert (n_mono.blocks[target_block_key_mono_save_test].Wq[0,1]==789.123).all(), "Monolith load fail (Wq value mismatch)"
        print("BandoRealityMeshMonolith save/load test PASSED (verified specific block state).")
    else:
        print("BandoRealityMeshMonolith save/load structure test PASSED (specific value check skipped as block was not suitable).")


    print("\n--- Testing MeshRouter ---")
    # Use the fol_tst mesh for the router
    router_mesh_primary_nodes = fol_tst.get_primary_nodes()
    num_test_nodes = len(router_mesh_primary_nodes) # Should be 7
    test_node_dim = dim_ex 
    test_models = []
    for i in range(num_test_nodes): # Create models for each of the 7 primary nodes
        if i % 3 == 0:
            test_models.append(VICtorchBlock(dim=test_node_dim, heads=2))
        elif i % 3 == 1:
            test_models.append(OmegaTensorBlock(dim=test_node_dim, tensor_order=2)) # Order 2 for simplicity
        else: 
            test_models.append(BandoBlock(dim=test_node_dim))

    router = MeshRouter(flower_of_life_mesh=fol_tst, 
                        node_models=test_models, 
                        k_iterations=2, 
                        attenuation=0.5)
    # Initial activations: list of vectors, one for each primary node of fol_tst
    initial_acts = [np.random.randn(test_node_dim) for _ in range(num_test_nodes)]
    final_acts = router.process(initial_activations=initial_acts)
    print(f"MeshRouter initial activation example shape: {initial_acts[0].shape if num_test_nodes > 0 else 'N/A'}")
    print(f"MeshRouter final activation example shape: {final_acts[0].shape if num_test_nodes > 0 and final_acts and final_acts[0] is not None else 'N/A'}")
    print(f"Number of final activations: {len(final_acts)}")
    assert len(final_acts) == num_test_nodes, "MeshRouter did not return correct number of activations."
    if num_test_nodes > 0 and final_acts and final_acts[0] is not None:
        assert final_acts[0].shape == (test_node_dim,), "MeshRouter output activation shape mismatch."
    print("MeshRouter basic processing test PASSED (structural checks).")

    print("\n--- Testing HeadCoordinatorBlock ---")
    # Using num_test_nodes (7 from fol_tst) and test_node_dim (dim_ex)
    input_dim_hcb = num_test_nodes * test_node_dim 
    hidden_dim_hcb = 128
    output_dim_hcb = test_node_dim # Output dim matches node model dim
    hcb = HeadCoordinatorBlock(dim=input_dim_hcb, hidden_dim=hidden_dim_hcb, output_dim=output_dim_hcb)
    dummy_fol_output = np.random.randn(input_dim_hcb) 
    final_response = hcb.forward(dummy_fol_output)
    print(f"HeadCoordinatorBlock input shape: {dummy_fol_output.shape}, output shape: {final_response.shape}")
    assert final_response.shape == (output_dim_hcb,), "HeadCoordinatorBlock output shape mismatch"
    hcb.W1[0,0] = 99.88
    hcb_state = hcb.get_state_dict()
    new_hcb = HeadCoordinatorBlock(dim=input_dim_hcb, hidden_dim=hidden_dim_hcb, output_dim=output_dim_hcb)
    new_hcb.load_state_dict(hcb_state)
    assert new_hcb.W1[0,0] == 99.88, "HeadCoordinatorBlock load_state_dict failed"
    print("HeadCoordinatorBlock save/load test PASSED.")

    print("\n--- Testing FlowerOfLifeNetworkOrchestrator Basic Save/Load ---")
    # Orchestrator uses its own mesh, distinct from fol_tst used for router test above
    orchestrator_nodes = 5 # Let's use a different number for orchestrator's internal mesh
    orchestrator_model_dim = dim_ex # 32
    fol_orchestrator = FlowerOfLifeNetworkOrchestrator(
        num_nodes=orchestrator_nodes, model_dim=orchestrator_model_dim, 
        mesh_depth=0, # Simpler mesh (just base)
        mesh_base_nodes=orchestrator_nodes, # Base nodes = num_nodes
        mesh_num_neighbors=2, 
        k_ripple_iterations=1, 
        coordinator_hidden_dim=64,
        coordinator_output_dim=orchestrator_model_dim 
    )
    # Check if num_nodes was adjusted by orchestrator based on mesh generation
    orchestrator_nodes = fol_orchestrator.num_nodes 
    print(f"Orchestrator initialized with {orchestrator_nodes} effective primary nodes.")

    fol_orchestrator.assign_block_to_node(0, "VICtorchBlock", heads=4)
    if orchestrator_nodes > 1: fol_orchestrator.assign_block_to_node(1, "OmegaTensorBlock")
    if orchestrator_nodes > 3: fol_orchestrator.assign_block_to_node(3, "FractalAttentionBlock", depth=1, heads=1) # Simpler Fractal
    
    print("Testing orchestrator process_input with single vector...")
    single_input_vector = np.random.randn(orchestrator_model_dim)
    response = fol_orchestrator.process_input(single_input_vector)
    if response is not None:
        print(f"Orchestrator response shape (single input): {response.shape}")
        assert response.shape == (orchestrator_model_dim,), "Orchestrator response shape mismatch for single input."
    else:
        print("Orchestrator process_input (single) returned None, check logs.")
    
    print("Testing orchestrator process_input with list of vectors...")
    # Create list matching the effective number of nodes
    list_input_vectors = [np.random.randn(orchestrator_model_dim) if i != 2 else None for i in range(orchestrator_nodes)] 
    response_list_input = fol_orchestrator.process_input(list_input_vectors)
    if response_list_input is not None:
        print(f"Orchestrator response shape (list input): {response_list_input.shape}")
        assert response_list_input.shape == (orchestrator_model_dim,), "Orchestrator response shape mismatch for list input."
    else:
        print("Orchestrator process_input (list) returned None, check logs.")

    orchestrator_save_path = "temp_fol_orchestrator_state.pkl"
    print(f"Saving orchestrator state to {orchestrator_save_path}...")
    fol_orchestrator.node_models[0].Wq[0,0] = 42.0 # Change state to check after load
    save_success = fol_orchestrator.save_network_state(orchestrator_save_path)
    assert save_success, "Failed to save orchestrator state."

    if save_success:
        print(f"Loading orchestrator state from {orchestrator_save_path}...")
        # Create a new orchestrator with default/dummy parameters, load_network_state should override them
        new_orchestrator = FlowerOfLifeNetworkOrchestrator(num_nodes=1, model_dim=10) 
        load_success = new_orchestrator.load_network_state(orchestrator_save_path)
        assert load_success, "Failed to load orchestrator state."

        if load_success:
            assert new_orchestrator.num_nodes == orchestrator_nodes, f"Loaded num_nodes mismatch: {new_orchestrator.num_nodes} vs {orchestrator_nodes}"
            assert new_orchestrator.model_dim == orchestrator_model_dim, "Loaded model_dim mismatch"
            assert new_orchestrator.node_models[0] is not None and isinstance(new_orchestrator.node_models[0], VICtorchBlock)
            assert (new_orchestrator.node_models[0].Wq[0,0] == 42.0).all(), "Loaded VICtorchBlock state mismatch"
            if orchestrator_nodes > 1: assert new_orchestrator.node_models[1] is not None and isinstance(new_orchestrator.node_models[1], OmegaTensorBlock)
            # Node 2 should be None as it wasn't assigned a block in the original orchestrator
            if orchestrator_nodes > 2: assert new_orchestrator.node_models[2] is None 
            if orchestrator_nodes > 3: assert new_orchestrator.node_models[3] is not None and isinstance(new_orchestrator.node_models[3], FractalAttentionBlock)
            
            print("Testing processing with loaded orchestrator...")
            response_after_load = new_orchestrator.process_input(single_input_vector)
            if response_after_load is not None:
                 print(f"Orchestrator response shape (after load): {response_after_load.shape}")
                 assert response_after_load.shape == (orchestrator_model_dim,)
            else:
                 print("Orchestrator process_input (after load) returned None.")
            print("FlowerOfLifeNetworkOrchestrator basic save/load and functionality test PASSED.")


    # --- Advanced Orchestrator Load Scenarios ---
    print("\n--- Testing FlowerOfLifeNetworkOrchestrator Advanced Load Scenarios ---")
    base_orchestrator_for_adv_tests_nodes = 3
    base_orchestrator_for_adv_tests_dim = 16 # Smaller dim for these tests
    adv_test_file = "temp_adv_orchestrator_state.pkl"

    # 1. Loading with an unknown block class name
    print("\n1. Test: Loading with an unknown block class name")
    orch1 = FlowerOfLifeNetworkOrchestrator(num_nodes=base_orchestrator_for_adv_tests_nodes, model_dim=base_orchestrator_for_adv_tests_dim, mesh_base_nodes=base_orchestrator_for_adv_tests_nodes)
    orch1.assign_block_to_node(0, "VICtorchBlock")
    # Manually create a state with an unknown block
    state1 = orch1.save_network_state(adv_test_file) # Save to get structure
    if state1:
        loaded_s1 = pickle.load(open(adv_test_file, "rb"))
        loaded_s1["node_model_states"][1] = {"class_name": "NonExistentBlock", "state_dict": {"dim": base_orchestrator_for_adv_tests_dim}}
        if base_orchestrator_for_adv_tests_nodes > 2 : loaded_s1["node_model_states"][2] = {"class_name": "BandoBlock", "state_dict": BandoBlock(dim=base_orchestrator_for_adv_tests_dim).get_state_dict()}
        else: # Ensure list is long enough if base_orchestrator_for_adv_tests_nodes was < 3
             while len(loaded_s1["node_model_states"]) < 3: loaded_s1["node_model_states"].append(None)
             loaded_s1["node_model_states"][2] = {"class_name": "BandoBlock", "state_dict": BandoBlock(dim=base_orchestrator_for_adv_tests_dim).get_state_dict()}

        pickle.dump(loaded_s1, open(adv_test_file, "wb"))

        orch1_loaded = FlowerOfLifeNetworkOrchestrator(num_nodes=1, model_dim=1) # Dummy orchestrator to load into
        orch1_loaded.load_network_state(adv_test_file)
        assert isinstance(orch1_loaded.node_models[0], VICtorchBlock), "Test 1 Failed: Valid block (VICtorch) not loaded."
        assert orch1_loaded.node_models[1] is None, "Test 1 Failed: Unknown block was not handled as None."
        if base_orchestrator_for_adv_tests_nodes > 2: assert isinstance(orch1_loaded.node_models[2], BandoBlock), "Test 1 Failed: Valid block (Bando) after unknown not loaded."
        print("Test 1 PASSED: Unknown block class handled gracefully.")
    else:
        print("Test 1 SKIPPED: Could not save initial state.")


    # 2. Loading a block state with missing 'dim' key
    print("\n2. Test: Loading a block state with missing 'dim' key")
    orch2 = FlowerOfLifeNetworkOrchestrator(num_nodes=base_orchestrator_for_adv_tests_nodes, model_dim=base_orchestrator_for_adv_tests_dim, mesh_base_nodes=base_orchestrator_for_adv_tests_nodes)
    orch2.assign_block_to_node(0, "VICtorchBlock") # Block whose state we'll modify
    state2 = orch2.save_network_state(adv_test_file)
    if state2:
        loaded_s2 = pickle.load(open(adv_test_file, "rb"))
        if loaded_s2["node_model_states"][0] and "state_dict" in loaded_s2["node_model_states"][0]:
            if "dim" in loaded_s2["node_model_states"][0]["state_dict"]:
                 del loaded_s2["node_model_states"][0]["state_dict"]["dim"] # Remove dim
            # Ensure other necessary keys like 'heads' for VICtorchBlock are present if its constructor needs them beyond 'dim'
            # The current load logic for VICtorchBlock in orchestrator gets 'heads' from state_dict too.
            # If 'dim' is missing, block_dim = state_dict.get("dim", self.model_dim) in load_network_state handles it.
        pickle.dump(loaded_s2, open(adv_test_file, "wb"))

        orch2_loaded = FlowerOfLifeNetworkOrchestrator(num_nodes=1, model_dim=base_orchestrator_for_adv_tests_dim+10) # Use a different model_dim for orchestrator
        orch2_loaded.load_network_state(adv_test_file)
        assert orch2_loaded.node_models[0] is not None, "Test 2 Failed: Block not loaded."
        # Dimension should default to the orchestrator's model_dim at time of load if not in state_dict
        # However, the orchestrator's model_dim itself gets updated from the *loaded network_state["model_dim"]* first.
        # So, the block's dim will be orch2.model_dim (base_orchestrator_for_adv_tests_dim)
        assert orch2_loaded.node_models[0].dim == base_orchestrator_for_adv_tests_dim, \
            f"Test 2 Failed: Block dim mismatch. Expected {base_orchestrator_for_adv_tests_dim}, Got {orch2_loaded.node_models[0].dim}"
        print("Test 2 PASSED: Missing 'dim' in block state handled (defaulted to network's model_dim from loaded state).")
    else:
        print("Test 2 SKIPPED: Could not save initial state.")

    # 3. Loading with different model_dim in the state
    print("\n3. Test: Loading state with different model_dim")
    orch3_orig_dim = base_orchestrator_for_adv_tests_dim # e.g. 16
    orch3_new_dim_in_orchestrator = orch3_orig_dim + 8 # e.g. 24
    orch3 = FlowerOfLifeNetworkOrchestrator(num_nodes=base_orchestrator_for_adv_tests_nodes, model_dim=orch3_orig_dim, mesh_base_nodes=base_orchestrator_for_adv_tests_nodes)
    orch3.assign_block_to_node(0, "BandoBlock") # Block with dim=orch3_orig_dim
    orch3.assign_block_to_node(1, "VICtorchBlock", dim=orch3_orig_dim, heads=2) # Explicit dim, heads
    # Save this state (model_dim will be orch3_orig_dim)
    state3 = orch3.save_network_state(adv_test_file)
    if state3:
        # Create new orchestrator with a *different* model_dim
        orch3_loaded = FlowerOfLifeNetworkOrchestrator(num_nodes=1, model_dim=orch3_new_dim_in_orchestrator)
        orch3_loaded.load_network_state(adv_test_file) # This should load model_dim from file (orch3_orig_dim)
        
        assert orch3_loaded.model_dim == orch3_orig_dim, \
            f"Test 3 Failed: Orchestrator model_dim not updated. Expected {orch3_orig_dim}, Got {orch3_loaded.model_dim}"
        assert orch3_loaded.node_models[0].dim == orch3_orig_dim, \
            f"Test 3 Failed: BandoBlock dim incorrect. Expected {orch3_orig_dim}, Got {orch3_loaded.node_models[0].dim}"
        assert orch3_loaded.node_models[1].dim == orch3_orig_dim, \
            f"Test 3 Failed: VICtorchBlock dim incorrect. Expected {orch3_orig_dim}, Got {orch3_loaded.node_models[1].dim}"
        print("Test 3 PASSED: Orchestrator model_dim updated from state; blocks use their respective/loaded dimensions.")
    else:
        print("Test 3 SKIPPED: Could not save initial state.")


    # 4. Loading with different mesh configuration
    print("\n4. Test: Loading state with different mesh configuration")
    orig_mesh_nodes = 3; orig_mesh_depth = 0; orig_model_dim = base_orchestrator_for_adv_tests_dim
    orch4 = FlowerOfLifeNetworkOrchestrator(num_nodes=orig_mesh_nodes, model_dim=orig_model_dim, 
                                           mesh_base_nodes=orig_mesh_nodes, mesh_depth=orig_mesh_depth, mesh_num_neighbors=2)
    orch4.assign_block_to_node(0, "BandoBlock") # Ensure at least one block
    state4 = orch4.save_network_state(adv_test_file) # Saves with mesh_base_nodes=3, depth=0
    if state4:
        # Create new orchestrator with different default mesh settings
        new_default_mesh_nodes = 5; new_default_mesh_depth = 1
        orch4_loaded = FlowerOfLifeNetworkOrchestrator(num_nodes=new_default_mesh_nodes, model_dim=orig_model_dim,
                                                     mesh_base_nodes=new_default_mesh_nodes, mesh_depth=new_default_mesh_depth)
        orch4_loaded.load_network_state(adv_test_file) # Load state with 3 nodes, depth 0

        assert orch4_loaded.mesh.base_nodes_count == orig_mesh_nodes, \
            f"Test 4 Failed: Mesh base_nodes mismatch. Expected {orig_mesh_nodes}, Got {orch4_loaded.mesh.base_nodes_count}"
        assert orch4_loaded.mesh.depth == orig_mesh_depth, \
            f"Test 4 Failed: Mesh depth mismatch. Expected {orig_mesh_depth}, Got {orch4_loaded.mesh.depth}"
        # num_nodes in orchestrator should be updated based on loaded mesh's primary nodes
        expected_num_nodes_after_load = len(orch4_loaded.mesh.get_primary_nodes())
        assert orch4_loaded.num_nodes == expected_num_nodes_after_load, \
             f"Test 4 Failed: Orchestrator num_nodes mismatch. Expected {expected_num_nodes_after_load}, Got {orch4_loaded.num_nodes}"
        # Also check if node_models list length matches
        assert len(orch4_loaded.node_models) == expected_num_nodes_after_load, \
             f"Test 4 Failed: node_models length mismatch. Expected {expected_num_nodes_after_load}, Got {len(orch4_loaded.node_models)}"
        # Check if the assigned block is still there (if new mesh config didn't make it impossible)
        if expected_num_nodes_after_load > 0 :
            assert isinstance(orch4_loaded.node_models[0], BandoBlock), "Test 4 Failed: Block assignment lost or incorrect after loading different mesh."
        else:
            print("Test 4 Warning: Loaded mesh has no primary nodes, block assignment check skipped.")

        print("Test 4 PASSED: Mesh configuration loaded correctly, orchestrator num_nodes and models list adjusted.")
    else:
        print("Test 4 SKIPPED: Could not save initial state.")


    # Cleanup temp files
    if os.path.exists(orchestrator_save_path):
        try: os.remove(orchestrator_save_path)
        except Exception as e_rem: print(f"Could not remove temp file {orchestrator_save_path}: {e_rem}")
    if os.path.exists(adv_test_file):
        try: os.remove(adv_test_file)
        except Exception as e_rem: print(f"Could not remove temp file {adv_test_file}: {e_rem}")
    if os.path.exists("temp_monolith_test.pkl"):
        try: os.remove("temp_monolith_test.pkl")
        except: pass
    
    print("\nAll tests complete.")
