Project Specification: TreeTNN Design (Final Version)
This document specifies the core data structure for TreeTNN in tensor4all-rs. Key Decision: To prioritize robustness and development velocity over premature memory optimization, the Graph Edges will store the full Index objects, not just lightweight IDs.

1. Core Concept
Node (Tensor): Wraps the calculation engine. It knows its own indices.

Edge (Connection): Represents the physical bond. It holds the actual Index objects defining the connection.

Update Strategy: When an operation (like SVD) creates new indices, we update the Tensor in the Node and replace the Index objects in the corresponding Edge.

2. Data Structures
A. Connection (The Edge)
Stores the explicit Index objects connecting two tensors.

Rust

use petgraph::graph::NodeIndex;

#[derive(Debug, Clone)]
pub struct Connection {
    // The exact Index objects used to bind the two tensors.
    // We store both sides to handle potential duality (bra/ket) explicitly 
    // and to validate matching dimensions easily.
    pub src_index: Index, 
    pub dst_index: Index,

    // Flow direction (Orthogonalization)
    pub ortho_towards: Option<NodeIndex>,
}

impl Connection {
    // Simple validation included in the constructor
    pub fn new(src: Index, dst: Index) -> Result<Self, String> {
        if src.dim != dst.dim {
            return Err(format!("Dimension mismatch: {} != {}", src.dim, dst.dim));
        }
        Ok(Self {
            src_index: src,
            dst_index: dst,
            ortho_towards: None,
        })
    }
}
B. TreeTNN (The Network)
Uses petgraph::StableGraph to maintain topology.

Rust

use petgraph::stable_graph::StableGraph;
use petgraph::Undirected;
use uuid::Uuid; // For TensorId

pub type TensorId = Uuid;

pub struct TreeTNN {
    // Nodes: TensorId (Handle to the tensor)
    // Edges: Connection (Stores the actual Indices)
    // Type: Undirected graph. Logic flow is handled by `ortho_towards`.
    pub graph: StableGraph<TensorId, Connection, Undirected>,
    
    // ... (TensorNode storage and lookups similar to previous iteration)
}