//! Tests for TreeTN operations: norm, norm_squared, inner, to_dense, evaluate.

use num_complex::Complex64;
use tensor4all_core::{
    AnyScalar, ColMajorArrayRef, DynIndex, IndexLike, TensorDynLen, TensorIndex,
};
use tensor4all_treetn::TreeTN;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a DynIndex with specific size.
fn idx(size: usize) -> DynIndex {
    DynIndex::new_dyn(size)
}

/// Create a TensorDynLen from indices and f64 data.
fn make_tensor(indices: Vec<DynIndex>, data: Vec<f64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn make_complex_tensor(indices: Vec<DynIndex>, data: Vec<Complex64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

/// Create a simple 2-node linear TreeTN with named nodes (usize).
/// Structure: node 0 (phys s0) -- bond -- node 1 (phys s1)
fn create_two_node_named() -> (
    TreeTN<TensorDynLen, usize>,
    DynIndex, // s0 (physical index at node 0)
    DynIndex, // bond
    DynIndex, // s1 (physical index at node 1)
) {
    let s0 = idx(2);
    let bond = idx(3);
    let s1 = idx(2);

    // t0 shape: [2, 3], t1 shape: [3, 2]
    let t0 = make_tensor(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    );
    let t1 = make_tensor(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );

    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();

    (tn, s0, bond, s1)
}

/// Create a 3-node chain TreeTN with named nodes.
/// Structure: node 0 -- bond01 -- node 1 -- bond12 -- node 2
fn create_three_node_named() -> (
    TreeTN<TensorDynLen, usize>,
    DynIndex, // s0
    DynIndex, // s1
    DynIndex, // s2
) {
    let s0 = idx(2);
    let bond01 = idx(3);
    let s1 = idx(2);
    let bond12 = idx(3);
    let s2 = idx(2);

    // Simple tensors with sequential data
    let dims0: usize = 2 * 3;
    let dims1: usize = 3 * 2 * 3;
    let dims2: usize = 3 * 2;

    let t0 = make_tensor(
        vec![s0.clone(), bond01.clone()],
        (0..dims0).map(|i| (i + 1) as f64).collect(),
    );
    let t1 = make_tensor(
        vec![bond01.clone(), s1.clone(), bond12.clone()],
        (0..dims1).map(|i| (i + 1) as f64 * 0.1).collect(),
    );
    let t2 = make_tensor(
        vec![bond12.clone(), s2.clone()],
        (0..dims2).map(|i| (i + 1) as f64).collect(),
    );

    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap();

    (tn, s0, s1, s2)
}

fn build_col_major_data(
    ordered_indices: &[DynIndex],
    point_values: &std::collections::HashMap<DynIndex, Vec<usize>>,
) -> Vec<usize> {
    let n_points = point_values.values().next().map(Vec::len).unwrap_or(0);
    let mut data = vec![0; ordered_indices.len() * n_points];

    for (row, index) in ordered_indices.iter().enumerate() {
        let per_index_values = point_values.get(index).unwrap();
        assert_eq!(per_index_values.len(), n_points);
        for (col, value) in per_index_values.iter().enumerate() {
            data[row + ordered_indices.len() * col] = *value;
        }
    }

    data
}

// ============================================================================
// Tests for norm and norm_squared
// ============================================================================

#[test]
fn test_norm_single_node() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![1.0, 2.0, 3.0]);
    let mut tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();

    let norm = tn.norm().unwrap();
    let expected = (1.0f64 + 4.0 + 9.0).sqrt();
    assert!(
        (norm - expected).abs() < 1e-10,
        "norm: got {}, expected {}",
        norm,
        expected
    );
}

#[test]
fn test_norm_squared_two_nodes() {
    let (mut tn, _, _, _) = create_two_node_named();

    let norm_sq = tn.norm_squared().unwrap();
    assert!(norm_sq > 0.0, "norm_squared must be positive");

    // Verify consistency: norm^2 == norm_squared
    let mut tn2 = tn.clone();
    let norm = tn2.norm().unwrap();
    assert!(
        (norm * norm - norm_sq).abs() / norm_sq < 1e-10,
        "norm^2 ({}) != norm_squared ({})",
        norm * norm,
        norm_sq
    );
}

#[test]
fn test_norm_against_dense() {
    let (mut tn, _, _, _) = create_three_node_named();

    let norm_tn = tn.norm().unwrap();

    // Compute norm from dense tensor
    let dense = tn.to_dense().unwrap();
    let norm_dense = dense.norm_squared().unwrap().sqrt();

    assert!(
        (norm_tn - norm_dense).abs() / norm_dense < 1e-10,
        "TreeTN norm ({}) != dense norm ({})",
        norm_tn,
        norm_dense
    );
}

// ============================================================================
// Tests for inner product
// ============================================================================

#[test]
fn test_inner_self() {
    let (tn, _, _, _) = create_two_node_named();

    let inner = tn.inner(&tn).unwrap();
    let mut tn2 = tn.clone();
    let norm_sq = tn2.norm_squared().unwrap();

    assert!(
        (inner.real() - norm_sq).abs() / norm_sq.max(1e-15) < 1e-10,
        "inner(self, self) ({}) != norm_squared ({})",
        inner.real(),
        norm_sq
    );
}

#[test]
fn test_inner_different_networks() {
    let s0 = idx(2);
    let bond = idx(3);
    let s1 = idx(2);

    let t0a = make_tensor(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    );
    let t1a = make_tensor(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
    let tn_a = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0a, t1a], vec![0, 1]).unwrap();

    let t0b = make_tensor(
        vec![s0.clone(), bond.clone()],
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    );
    let t1b = make_tensor(
        vec![bond.clone(), s1.clone()],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    );
    let tn_b = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0b, t1b], vec![0, 1]).unwrap();

    // Compute inner product via TreeTN method
    let inner_val = tn_a.inner(&tn_b).unwrap();

    // Compute expected via dense tensors
    let dense_a = tn_a.to_dense().unwrap();
    let dense_b = tn_b.to_dense().unwrap();
    let expected = dense_a.inner_product(&dense_b).unwrap();

    assert!(
        (inner_val.real() - expected.real()).abs() < 1e-10,
        "inner({}, {}) != expected ({})",
        inner_val.real(),
        expected.real(),
        expected.real()
    );
}

#[test]
fn test_inner_three_nodes() {
    let (tn, _, _, _) = create_three_node_named();

    let inner = tn.inner(&tn).unwrap();
    let mut tn2 = tn.clone();
    let norm_sq = tn2.norm_squared().unwrap();

    assert!(
        (inner.real() - norm_sq).abs() / norm_sq.max(1e-15) < 1e-10,
        "inner(self, self) ({}) != norm_squared ({})",
        inner.real(),
        norm_sq
    );
}

// ============================================================================
// Tests for to_dense
// ============================================================================

#[test]
fn test_to_dense_single_node() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![1.0, 2.0, 3.0]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0.clone()], vec![0]).unwrap();

    let dense = tn.to_dense().unwrap();
    assert!(
        dense.isapprox(&t0, 1e-10, 0.0).unwrap(),
        "to_dense mismatch: maxabs diff = {}",
        dense.distance(&t0).unwrap()
    );
}

#[test]
fn test_to_dense_two_nodes() {
    let (tn, _, _, _) = create_two_node_named();
    let dense = tn.to_dense().unwrap();

    // Result should have only site indices (bond contracted)
    assert_eq!(dense.external_indices().len(), 2);
}

#[test]
fn test_to_dense_empty() {
    let tn = TreeTN::<TensorDynLen, usize>::new();
    let result = tn.to_dense();
    assert!(result.is_err());
}

// ============================================================================
// Tests for all_site_indices
// ============================================================================

#[test]
fn test_all_site_indices_single_node_returns_indices() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![1.0, 2.0, 3.0]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();

    let (indices, vertices) = tn.all_site_indices().unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(vertices.len(), 1);
    assert_eq!(indices[0], s0);
    assert_eq!(vertices[0], 0);
}

#[test]
fn test_all_site_indices_two_nodes_returns_indices() {
    let (tn, s0, _bond, s1) = create_two_node_named();

    let (indices, vertices) = tn.all_site_indices().unwrap();
    assert_eq!(indices.len(), 2);
    assert_eq!(vertices.len(), 2);

    // Check that both site indices are present with correct vertex associations
    let index_vertex_set: std::collections::HashSet<_> =
        indices.iter().zip(vertices.iter()).collect();
    assert!(index_vertex_set.contains(&(&s0, &0)));
    assert!(index_vertex_set.contains(&(&s1, &1)));
}

#[test]
fn test_all_site_indices_three_nodes_returns_indices() {
    let (tn, s0, s1, s2) = create_three_node_named();

    let (indices, vertices) = tn.all_site_indices().unwrap();
    assert_eq!(indices.len(), 3);
    assert_eq!(vertices.len(), 3);

    let index_vertex_set: std::collections::HashSet<_> =
        indices.iter().zip(vertices.iter()).collect();
    assert!(index_vertex_set.contains(&(&s0, &0)));
    assert!(index_vertex_set.contains(&(&s1, &1)));
    assert!(index_vertex_set.contains(&(&s2, &2)));
}

// ============================================================================
// Tests for evaluate
// ============================================================================

#[test]
fn test_evaluate_single_node_with_dense() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![10.0, 20.0, 30.0]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();

    // Evaluate at index 0
    let data = [0usize];
    let shape = [indices.len(), 1];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let vals = tn.evaluate(&indices, values).unwrap();
    assert!(
        (vals[0].real() - 10.0).abs() < 1e-10,
        "evaluate at [0] = {}, expected 10.0",
        vals[0].real()
    );

    // Evaluate at index 2
    let data = [2usize];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let vals = tn.evaluate(&indices, values).unwrap();
    assert!(
        (vals[0].real() - 30.0).abs() < 1e-10,
        "evaluate at [2] = {}, expected 30.0",
        vals[0].real()
    );
}

#[test]
fn test_evaluate_two_nodes_sugar() {
    let (tn, s0, _, s1) = create_two_node_named();

    // Get the dense representation for reference
    let dense = tn.to_dense().unwrap();
    let dense_data = dense.to_vec::<f64>().unwrap();

    let dim0 = s0.dim();
    let dim1 = s1.dim();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    // Find positions of s0 and s1 in indices
    let pos0 = indices.iter().position(|index| index == &s0).unwrap();
    let pos1 = indices.iter().position(|index| index == &s1).unwrap();

    // Evaluate at each combination and verify against dense
    for i in 0..dim0 {
        for j in 0..dim1 {
            let mut data = vec![0usize; indices.len()];
            data[pos0] = i;
            data[pos1] = j;
            let shape = [indices.len(), 1];
            let values = ColMajorArrayRef::new(&data, &shape).unwrap();
            let vals = tn.evaluate(&indices, values).unwrap();

            // Get the expected value from dense tensor
            let flat_idx = i + dim0 * j;
            let expected = dense_data[flat_idx];

            assert!(
                (vals[0].real() - expected).abs() < 1e-10,
                "evaluate at [{}, {}] = {}, expected {}",
                i,
                j,
                vals[0].real(),
                expected
            );
        }
    }
}

#[test]
fn test_evaluate_two_nodes_complex() {
    let s0 = idx(2);
    let bond = idx(1);
    let s1 = idx(2);
    let a = [Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)];
    let b = [Complex64::new(3.0, -2.0), Complex64::new(-1.0, 4.0)];

    let t0 = make_complex_tensor(vec![s0.clone(), bond.clone()], vec![a[0], a[1]]);
    let t1 = make_complex_tensor(vec![bond.clone(), s1.clone()], vec![b[0], b[1]]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos0 = indices.iter().position(|index| index == &s0).unwrap();
    let pos1 = indices.iter().position(|index| index == &s1).unwrap();

    let max_sample = a
        .iter()
        .flat_map(|a_i| b.iter().map(move |b_j| (*a_i * *b_j).norm()))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let mut max_diff = 0.0_f64;
    for (i, a_i) in a.iter().enumerate() {
        for (j, b_j) in b.iter().enumerate() {
            let mut data = vec![0usize; indices.len()];
            data[pos0] = i;
            data[pos1] = j;
            let shape = [indices.len(), 1];
            let values = ColMajorArrayRef::new(&data, &shape).unwrap();
            let vals = tn.evaluate(&indices, values).unwrap();
            let expected = *a_i * *b_j;
            let got = Complex64::new(vals[0].real(), vals[0].imag());
            max_diff = max_diff.max((got - expected).norm());
        }
    }
    assert!(
        max_diff <= 1e-12 * max_sample,
        "maxabs diff {} exceeds tol {} * max_sample {}",
        max_diff,
        1e-12,
        max_sample
    );
}

#[test]
fn test_evaluate_three_nodes_sugar() {
    let (tn, s0, s1, s2) = create_three_node_named();

    // Get dense for reference
    let dense = tn.to_dense().unwrap();
    let dense_data = dense.to_vec::<f64>().unwrap();

    let dim0 = s0.dim();
    let dim1 = s1.dim();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos0 = indices.iter().position(|index| index == &s0).unwrap();
    let pos1 = indices.iter().position(|index| index == &s1).unwrap();
    let pos2 = indices.iter().position(|index| index == &s2).unwrap();

    // Spot-check a few values
    for (i, j, k) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] {
        let mut data = vec![0usize; indices.len()];
        data[pos0] = i;
        data[pos1] = j;
        data[pos2] = k;
        let shape = [indices.len(), 1];
        let values = ColMajorArrayRef::new(&data, &shape).unwrap();
        let vals = tn.evaluate(&indices, values).unwrap();

        let flat_idx = i + dim0 * (j + dim1 * k);
        let expected = dense_data[flat_idx];

        assert!(
            (vals[0].real() - expected).abs() < 1e-8,
            "evaluate at [{}, {}, {}] = {}, expected {}",
            i,
            j,
            k,
            vals[0].real(),
            expected
        );
    }
}

#[test]
fn test_evaluate_matches_batch_evaluate() {
    let (tn, s0, s1, s2) = create_three_node_named();

    let (indices, _node_names) = tn.all_site_indices().unwrap();

    let point_values = std::collections::HashMap::from([
        (s0.clone(), vec![0usize, 1, 1]),
        (s1.clone(), vec![0usize, 1, 0]),
        (s2.clone(), vec![1usize, 0, 1]),
    ]);

    let shape = [indices.len(), 3];

    let evaluate_data = build_col_major_data(&indices, &point_values);
    let evaluate_values = ColMajorArrayRef::new(&evaluate_data, &shape).unwrap();
    let evaluate_result = tn.evaluate(&indices, evaluate_values).unwrap();

    let evaluate_data_2 = build_col_major_data(&indices, &point_values);
    let evaluate_values_2 = ColMajorArrayRef::new(&evaluate_data_2, &shape).unwrap();
    let evaluate_result_2 = tn.evaluate(&indices, evaluate_values_2).unwrap();

    assert_eq!(evaluate_result_2.len(), evaluate_result.len());
    for (evaluate_value_2, evaluate_value) in evaluate_result_2.iter().zip(evaluate_result.iter()) {
        assert!((evaluate_value_2.real() - evaluate_value.real()).abs() < 1e-12);
        assert!((evaluate_value_2.imag() - evaluate_value.imag()).abs() < 1e-12);
    }
}

#[test]
fn test_evaluate_empty() {
    let tn = TreeTN::<TensorDynLen, usize>::new();
    let data: [usize; 0] = [];
    let shape = [0, 1];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let result = tn.evaluate(&[], values);
    assert!(result.is_err());
}

// ============================================================================
// Tests for evaluate validation
// ============================================================================

#[test]
fn test_evaluate_rejects_duplicate_indices() {
    let (tn, s0, _, _s1) = create_two_node_named();

    // Use the same index twice instead of both distinct site indices.
    let dup_indices = vec![s0.clone(), s0.clone()];
    let data = [0usize, 0];
    let shape = [2, 1];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let result = tn.evaluate(&dup_indices, values);
    assert!(result.is_err(), "should reject duplicate indices");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("duplicate"),
        "error should mention 'duplicate', got: {msg}"
    );
}

#[test]
fn test_evaluate_rejects_unknown_indices() {
    let (tn, s0, _, _s1) = create_two_node_named();

    // Use one real index and one fabricated index that does not exist in the network.
    let unknown = idx(2);
    let fake_indices = vec![s0.clone(), unknown];
    let data = [0usize, 0];
    let shape = [2, 1];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let result = tn.evaluate(&fake_indices, values);
    assert!(result.is_err(), "should reject unknown indices");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("unknown"),
        "error should mention 'unknown', got: {msg}"
    );
}

#[test]
fn test_evaluate_rejects_missing_indices() {
    let (tn, s0, _, _s1) = create_two_node_named();

    // Provide only one of the two required site indices.
    let partial_indices = vec![s0.clone()];
    let data = [0usize];
    let shape = [1, 1];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let result = tn.evaluate(&partial_indices, values);
    assert!(result.is_err(), "should reject missing indices");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("total site indices"),
        "error should mention count mismatch, got: {msg}"
    );
}

// ============================================================================
// Tests for all_site_indices
// ============================================================================

#[test]
fn test_all_site_indices_single_node() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![1.0, 2.0, 3.0]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();

    let (indices, vertices) = tn.all_site_indices().unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(vertices.len(), 1);
    assert_eq!(*indices[0].id(), *s0.id());
    assert_eq!(vertices[0], 0);
}

#[test]
fn test_all_site_indices_two_nodes() {
    let (tn, s0, _bond, s1) = create_two_node_named();

    let (indices, vertices) = tn.all_site_indices().unwrap();
    assert_eq!(indices.len(), 2);
    assert_eq!(vertices.len(), 2);

    // Check that both site indices are present with correct vertex associations
    let idx_vertex_set: std::collections::HashSet<_> = indices
        .iter()
        .map(|i| *i.id())
        .zip(vertices.iter())
        .collect();
    assert!(idx_vertex_set.contains(&(*s0.id(), &0)));
    assert!(idx_vertex_set.contains(&(*s1.id(), &1)));
}

#[test]
fn test_all_site_indices_has_matching_vertices() {
    let (tn, _, _, _) = create_three_node_named();

    let (indices, idx_vertices) = tn.all_site_indices().unwrap();

    assert_eq!(indices.len(), idx_vertices.len());

    // Build a set of (index, vertex) pairs to ensure ownership is represented.
    let idx_set: std::collections::HashSet<_> = indices
        .iter()
        .cloned()
        .zip(idx_vertices.iter().cloned())
        .collect();

    assert_eq!(idx_set.len(), 3);
}

// ============================================================================
// Tests for evaluate
// ============================================================================

#[test]
fn test_evaluate_single_node_sugar() {
    let s0 = idx(3);
    let t0 = make_tensor(vec![s0.clone()], vec![10.0, 20.0, 30.0]);
    let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();

    let (indices, _vertices) = tn.all_site_indices().unwrap();

    // Evaluate at index 0
    let data = [0usize];
    let shape = [indices.len(), 1];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let vals = tn.evaluate(&indices, values).unwrap();
    assert!(
        (vals[0].real() - 10.0).abs() < 1e-10,
        "evaluate at [0] = {}, expected 10.0",
        vals[0].real()
    );

    // Evaluate at index 2
    let data = [2usize];
    let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    let vals = tn.evaluate(&indices, values).unwrap();
    assert!(
        (vals[0].real() - 30.0).abs() < 1e-10,
        "evaluate at [2] = {}, expected 30.0",
        vals[0].real()
    );
}

#[test]
fn test_evaluate_two_nodes() {
    let (tn, s0, _, s1) = create_two_node_named();

    // Get the dense representation for reference
    let dense = tn.to_dense().unwrap();
    let dense_data = dense.to_vec::<f64>().unwrap();

    let dim0 = s0.dim();
    let dim1 = s1.dim();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    // Find positions of s0 and s1 in indices
    let pos0 = indices.iter().position(|i| *i.id() == *s0.id()).unwrap();
    let pos1 = indices.iter().position(|i| *i.id() == *s1.id()).unwrap();

    // Evaluate at each combination and verify against dense
    for i in 0..dim0 {
        for j in 0..dim1 {
            let mut data = vec![0usize; indices.len()];
            data[pos0] = i;
            data[pos1] = j;
            let shape = [indices.len(), 1];
            let values = ColMajorArrayRef::new(&data, &shape).unwrap();
            let vals = tn.evaluate(&indices, values).unwrap();

            let flat_idx = i + dim0 * j;
            let expected = dense_data[flat_idx];

            assert!(
                (vals[0].real() - expected).abs() < 1e-10,
                "evaluate at [{}, {}] = {}, expected {}",
                i,
                j,
                vals[0].real(),
                expected
            );
        }
    }
}

#[test]
fn test_evaluate_three_nodes() {
    let (tn, s0, s1, s2) = create_three_node_named();

    let dense = tn.to_dense().unwrap();
    let dense_data = dense.to_vec::<f64>().unwrap();

    let dim0 = s0.dim();
    let dim1 = s1.dim();

    let (indices, _vertices) = tn.all_site_indices().unwrap();
    let pos0 = indices.iter().position(|i| *i.id() == *s0.id()).unwrap();
    let pos1 = indices.iter().position(|i| *i.id() == *s1.id()).unwrap();
    let pos2 = indices.iter().position(|i| *i.id() == *s2.id()).unwrap();

    for (i, j, k) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)] {
        let mut data = vec![0usize; indices.len()];
        data[pos0] = i;
        data[pos1] = j;
        data[pos2] = k;
        let shape = [indices.len(), 1];
        let values = ColMajorArrayRef::new(&data, &shape).unwrap();
        let vals = tn.evaluate(&indices, values).unwrap();

        let flat_idx = i + dim0 * (j + dim1 * k);
        let expected = dense_data[flat_idx];

        assert!(
            (vals[0].real() - expected).abs() < 1e-8,
            "evaluate at [{}, {}, {}] = {}, expected {}",
            i,
            j,
            k,
            vals[0].real(),
            expected
        );
    }
}

// ============================================================================
// Tests for add (which already exists but we test the public API)
// ============================================================================

#[test]
fn test_add_two_nodes() {
    let (tn_a, _, _, _) = create_two_node_named();
    let tn_b = tn_a.clone();

    let sum = tn_a.add(&tn_b).unwrap();

    // Verify numerically: sum.to_dense() == 2 * tn_a.to_dense()
    let sum_dense = sum.to_dense().unwrap();
    let a_dense = tn_a.to_dense().unwrap();
    let expected = a_dense.scale(AnyScalar::new_real(2.0)).unwrap();
    assert!(
        sum_dense.isapprox(&expected, 1e-10, 0.0).unwrap(),
        "add mismatch: maxabs diff = {}",
        sum_dense.distance(&expected).unwrap()
    );
}
