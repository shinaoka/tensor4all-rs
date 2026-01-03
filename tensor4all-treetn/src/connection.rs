use tensor4all::index::{Index, Symmetry};
use tensor4all::DefaultTagSet;
use anyhow::{Result, Context};

/// Connection between two tensors in a tree tensor network.
///
/// Stores the Index objects on both sides of the bond. For undirected graphs,
/// the source/target distinction is determined by the edge direction in petgraph,
/// but both indices are stored explicitly.
#[derive(Debug, Clone)]
pub struct Connection<Id, Symm, Tags = DefaultTagSet> {
    /// Index on the source side of the edge (as determined by petgraph's edge direction).
    pub index_source: Index<Id, Symm, Tags>,
    /// Index on the target side of the edge (as determined by petgraph's edge direction).
    pub index_target: Index<Id, Symm, Tags>,
    /// Optional orthogonalization direction.
    /// When Some(index), indicates that orthogonalization flows towards the node
    /// that has this index. The index must be either index_source or index_target.
    pub ortho_towards: Option<Index<Id, Symm, Tags>>,
}

impl<Id, Symm, Tags> Connection<Id, Symm, Tags>
where
    Id: std::hash::Hash + Eq + Clone,
    Symm: Symmetry + Clone,
    Tags: Default + Clone,
{
    /// Create a new connection between two indices.
    ///
    /// Validates that the indices have matching dimensions.
    pub fn new(
        index_source: Index<Id, Symm, Tags>,
        index_target: Index<Id, Symm, Tags>,
    ) -> Result<Self> {
        // Validate dimension matching
        if index_source.size() != index_target.size() {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: {} != {}",
                index_source.size(),
                index_target.size()
            ))
            .context("Failed to create connection: indices must have matching dimensions");
        }

        Ok(Self {
            index_source,
            index_target,
            ortho_towards: None,
        })
    }

    /// Create a connection with orthogonalization direction.
    ///
    /// The ortho_towards index must be either index_source or index_target.
    pub fn with_ortho_direction(
        index_source: Index<Id, Symm, Tags>,
        index_target: Index<Id, Symm, Tags>,
        ortho_towards: Index<Id, Symm, Tags>,
    ) -> Result<Self> {
        let mut conn = Self::new(index_source.clone(), index_target.clone())?;
        
        // Validate that ortho_towards is one of the connection indices
        if ortho_towards.id != index_source.id && ortho_towards.id != index_target.id {
            return Err(anyhow::anyhow!(
                "ortho_towards index must be either index_source or index_target"
            ))
            .context("with_ortho_direction: invalid ortho_towards index");
        }
        
        conn.ortho_towards = Some(ortho_towards);
        Ok(conn)
    }

    /// Get the bond dimension (size of the indices).
    pub fn bond_dim(&self) -> usize {
        self.index_source.size()
    }

    /// Set the orthogonalization direction.
    ///
    /// The index must be either index_source or index_target (or None).
    pub fn set_ortho_towards(
        &mut self,
        ortho_towards: Option<Index<Id, Symm, Tags>>,
    ) -> Result<()> {
        if let Some(ref ortho_idx) = ortho_towards {
            if ortho_idx.id != self.index_source.id && ortho_idx.id != self.index_target.id {
                return Err(anyhow::anyhow!(
                    "ortho_towards index must be either index_source or index_target"
                ))
                .context("set_ortho_towards: invalid ortho_towards index");
            }
        }
        self.ortho_towards = ortho_towards;
        Ok(())
    }

    /// Replace the bond indices (e.g., after SVD creates new bond indices).
    ///
    /// Validates that the new indices have matching dimensions.
    pub fn replace_bond_indices(
        &mut self,
        new_index_source: Index<Id, Symm, Tags>,
        new_index_target: Index<Id, Symm, Tags>,
    ) -> Result<()> {
        if new_index_source.size() != new_index_target.size() {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: {} != {}",
                new_index_source.size(),
                new_index_target.size()
            ))
            .context("Failed to replace bond indices: new indices must have matching dimensions");
        }
        self.index_source = new_index_source;
        self.index_target = new_index_target;
        Ok(())
    }
}

