use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, ensure, Result};
use num_complex::Complex64;
use tenferro::{DType, Tensor as NativeTensor};
use tensor4all_tensorbackend::{einsum_native_tensors, NativeTensorReadInput, Storage};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperandLayout {
    pub(crate) logical_dims: Vec<usize>,
    pub(crate) axis_classes: Vec<usize>,
}

impl OperandLayout {
    pub(crate) fn new(logical_dims: Vec<usize>, axis_classes: Vec<usize>) -> Result<Self> {
        ensure!(
            logical_dims.len() == axis_classes.len(),
            "logical_dims rank {} does not match axis_classes rank {}",
            logical_dims.len(),
            axis_classes.len()
        );
        validate_axis_classes(&axis_classes)?;

        let mut dims_by_class = HashMap::new();
        for (&dim, &class_id) in logical_dims.iter().zip(axis_classes.iter()) {
            if let Some(&expected) = dims_by_class.get(&class_id) {
                ensure!(
                    expected == dim,
                    "axis class {class_id} has inconsistent dimensions {expected} and {dim}"
                );
            } else {
                dims_by_class.insert(class_id, dim);
            }
        }

        Ok(Self {
            logical_dims,
            axis_classes,
        })
    }

    fn payload_rank(&self) -> usize {
        self.axis_classes
            .iter()
            .copied()
            .max()
            .map(|value| value + 1)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StructuredContractionSpec {
    pub(crate) input_labels: Vec<Vec<usize>>,
    pub(crate) output_labels: Vec<usize>,
    pub(crate) retained_labels: HashSet<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperandPayloadPlan {
    pub(crate) class_roots: Vec<usize>,
    pub(crate) normalized_class_roots: Vec<usize>,
}

impl OperandPayloadPlan {
    #[cfg(test)]
    pub(crate) fn has_repeated_roots(&self) -> bool {
        self.class_roots.len() != self.normalized_class_roots.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StructuredContractionPlan {
    pub(crate) operand_plans: Vec<OperandPayloadPlan>,
    pub(crate) output_axis_classes: Vec<usize>,
    pub(crate) output_payload_roots: Vec<usize>,
    pub(crate) output_payload_dims: Vec<usize>,
}

impl StructuredContractionPlan {
    pub(crate) fn new(
        operands: &[OperandLayout],
        spec: &StructuredContractionSpec,
    ) -> Result<Self> {
        ensure!(
            operands.len() == spec.input_labels.len(),
            "operand count {} does not match input label count {}",
            operands.len(),
            spec.input_labels.len()
        );

        let offsets = node_offsets(operands);
        let total_nodes = offsets.last().copied().unwrap_or(0);
        let mut uf = UnionFind::new(total_nodes);
        let mut label_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut label_dims: HashMap<usize, usize> = HashMap::new();

        for (operand_idx, operand) in operands.iter().enumerate() {
            let labels = &spec.input_labels[operand_idx];
            ensure!(
                labels.len() == operand.logical_dims.len(),
                "input label rank {} does not match operand {operand_idx} rank {}",
                labels.len(),
                operand.logical_dims.len()
            );

            for (axis, (&label, &dim)) in labels.iter().zip(operand.logical_dims.iter()).enumerate()
            {
                if let Some(&expected) = label_dims.get(&label) {
                    ensure!(
                        expected == dim,
                        "label {label} has inconsistent dimensions {expected} and {dim}"
                    );
                } else {
                    label_dims.insert(label, dim);
                }

                let class_id = operand.axis_classes[axis];
                let node = offsets[operand_idx] + class_id;
                label_nodes.entry(label).or_default().push(node);
            }
        }

        for nodes in label_nodes.values() {
            if let Some((&first, rest)) = nodes.split_first() {
                for &node in rest {
                    uf.union(first, node);
                }
            }
        }

        let node_roots = canonical_roots(&mut uf, total_nodes);
        let root_dims = root_dimensions(operands, &offsets, &node_roots)?;

        let mut operand_plans = Vec::with_capacity(operands.len());
        for (operand_idx, operand) in operands.iter().enumerate() {
            let class_roots: Vec<usize> = (0..operand.payload_rank())
                .map(|class_id| node_roots[offsets[operand_idx] + class_id])
                .collect();
            let normalized_class_roots = unique_first_appearance(&class_roots);
            operand_plans.push(OperandPayloadPlan {
                class_roots,
                normalized_class_roots,
            });
        }

        let mut output_roots = Vec::with_capacity(spec.output_labels.len());
        for &label in &spec.output_labels {
            let nodes = label_nodes
                .get(&label)
                .ok_or_else(|| anyhow!("output label {label} is not present in inputs"))?;
            output_roots.push(node_roots[nodes[0]]);
        }

        let output_axis_classes = canonicalize_sequence(&output_roots);
        let output_payload_roots = unique_first_appearance(&output_roots);
        let output_payload_dims = output_payload_roots
            .iter()
            .map(|root| root_dims[*root])
            .collect();

        // Future retained-label options are interpreted by choosing output_labels.
        // Reading the set here keeps the field semantically live.
        for retained in &spec.retained_labels {
            ensure!(
                label_nodes.contains_key(retained),
                "retained label {retained} is not present in inputs"
            );
        }

        Ok(Self {
            operand_plans,
            output_axis_classes,
            output_payload_roots,
            output_payload_dims,
        })
    }
}

fn validate_axis_classes(axis_classes: &[usize]) -> Result<()> {
    let mut next = 0usize;
    for &class_id in axis_classes {
        ensure!(
            class_id <= next,
            "axis_classes must be canonical first-appearance labels, got {axis_classes:?}"
        );
        if class_id == next {
            next += 1;
        }
    }
    Ok(())
}

fn node_offsets(operands: &[OperandLayout]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(operands.len() + 1);
    let mut next = 0usize;
    offsets.push(next);
    for operand in operands {
        next += operand.payload_rank();
        offsets.push(next);
    }
    offsets
}

fn canonical_roots(uf: &mut UnionFind, total_nodes: usize) -> Vec<usize> {
    let mut root_to_id = HashMap::new();
    let mut next_id = 0usize;
    (0..total_nodes)
        .map(|node| {
            let root = uf.find(node);
            *root_to_id.entry(root).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect()
}

fn root_dimensions(
    operands: &[OperandLayout],
    offsets: &[usize],
    node_roots: &[usize],
) -> Result<Vec<usize>> {
    let mut dims_by_root = HashMap::new();
    for (operand_idx, operand) in operands.iter().enumerate() {
        for (axis, &dim) in operand.logical_dims.iter().enumerate() {
            let class_id = operand.axis_classes[axis];
            let root = node_roots[offsets[operand_idx] + class_id];
            if let Some(&expected) = dims_by_root.get(&root) {
                ensure!(
                    expected == dim,
                    "merged root {root} has inconsistent dimensions {expected} and {dim}"
                );
            } else {
                dims_by_root.insert(root, dim);
            }
        }
    }

    let len = dims_by_root
        .keys()
        .copied()
        .max()
        .map(|root| root + 1)
        .unwrap_or(0);
    let mut dims = vec![1; len];
    for (root, dim) in dims_by_root {
        dims[root] = dim;
    }
    Ok(dims)
}

fn unique_first_appearance(values: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for &value in values {
        if seen.insert(value) {
            result.push(value);
        }
    }
    result
}

#[allow(dead_code)]
pub(crate) fn normalize_payload_for_roots(
    payload: &NativeTensor,
    roots: &[usize],
) -> Result<(NativeTensor, Vec<usize>)> {
    ensure!(
        payload.shape().len() == roots.len(),
        "payload rank {} does not match root label count {}",
        payload.shape().len(),
        roots.len()
    );

    if unique_first_appearance(roots).len() == roots.len() {
        return Ok((payload.clone(), roots.to_vec()));
    }

    let mut current_payload = payload.clone();
    let mut current_roots = roots.to_vec();
    while let Some((axis_a, axis_b)) = first_duplicate_pair(&current_roots) {
        let mut input_ids: Vec<usize> = (0..current_roots.len()).collect();
        input_ids[axis_b] = input_ids[axis_a];
        let output_ids: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(axis, &label)| (axis != axis_b).then_some(label))
            .collect();

        current_payload = einsum_native_tensors(&[(&current_payload, &input_ids)], &output_ids)?;
        current_roots.remove(axis_b);
    }

    Ok((current_payload, current_roots))
}

pub(crate) fn normalize_payload_read_for_roots<'a>(
    payload: NativeTensorReadInput<'a>,
    roots: &[usize],
) -> Result<(NativeTensorReadInput<'a>, Vec<usize>)> {
    ensure!(
        payload.shape().len() == roots.len(),
        "payload rank {} does not match root label count {}",
        payload.shape().len(),
        roots.len()
    );

    if unique_first_appearance(roots).len() == roots.len() {
        return Ok((payload, roots.to_vec()));
    }

    let mut current_payload = payload.as_read().to_tensor();
    let mut current_roots = roots.to_vec();
    while let Some((axis_a, axis_b)) = first_duplicate_pair(&current_roots) {
        let mut input_ids: Vec<usize> = (0..current_roots.len()).collect();
        input_ids[axis_b] = input_ids[axis_a];
        let output_ids: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(axis, &label)| (axis != axis_b).then_some(label))
            .collect();

        current_payload = einsum_native_tensors(&[(&current_payload, &input_ids)], &output_ids)?;
        current_roots.remove(axis_b);
    }

    Ok((NativeTensorReadInput::Owned(current_payload), current_roots))
}

pub(crate) fn storage_payload_native(storage: &Storage) -> Result<NativeTensor> {
    if storage.is_f64() {
        Ok(NativeTensor::from_vec_col_major(
            storage.payload_dims().to_vec(),
            storage
                .payload_f64_col_major_vec()
                .map_err(anyhow::Error::new)?,
        ))
    } else if storage.is_c64() {
        Ok(NativeTensor::from_vec_col_major(
            storage.payload_dims().to_vec(),
            storage
                .payload_c64_col_major_vec()
                .map_err(anyhow::Error::new)?,
        ))
    } else {
        Err(anyhow!("unsupported storage scalar type"))
    }
}

pub(crate) fn storage_from_payload_native(
    payload: NativeTensor,
    output_payload_dims: &[usize],
    output_axis_classes: Vec<usize>,
) -> Result<Storage> {
    ensure!(
        payload.shape() == output_payload_dims,
        "payload shape {:?} does not match planned payload dims {:?}",
        payload.shape(),
        output_payload_dims
    );
    let strides = col_major_strides(output_payload_dims)?;
    match payload.dtype() {
        DType::F64 => {
            let values = payload
                .as_slice::<f64>()
                .ok_or_else(|| anyhow!("failed to read f64 payload tensor"))?
                .to_vec();
            Storage::new_structured(
                values,
                output_payload_dims.to_vec(),
                strides,
                output_axis_classes,
            )
        }
        DType::C64 => {
            let values = payload
                .as_slice::<Complex64>()
                .ok_or_else(|| anyhow!("failed to read Complex64 payload tensor"))?
                .to_vec();
            Storage::new_structured(
                values,
                output_payload_dims.to_vec(),
                strides,
                output_axis_classes,
            )
        }
        dtype => Err(anyhow!(
            "unsupported structured payload dtype {dtype:?}; expected f64 or Complex64"
        )),
    }
}

fn col_major_strides(dims: &[usize]) -> Result<Vec<isize>> {
    let mut stride = 1isize;
    let mut strides = Vec::with_capacity(dims.len());
    for &dim in dims {
        strides.push(stride);
        stride = stride
            .checked_mul(dim as isize)
            .ok_or_else(|| anyhow!("payload stride overflow for dims {dims:?}"))?;
    }
    Ok(strides)
}

fn first_duplicate_pair(values: &[usize]) -> Option<(usize, usize)> {
    let mut first_axis_by_value = HashMap::new();
    for (axis, &value) in values.iter().enumerate() {
        if let Some(&first_axis) = first_axis_by_value.get(&value) {
            return Some((first_axis, axis));
        }
        first_axis_by_value.insert(value, axis);
    }
    None
}

fn canonicalize_sequence(values: &[usize]) -> Vec<usize> {
    let mut id_by_value = HashMap::new();
    let mut next = 0usize;
    values
        .iter()
        .map(|value| {
            *id_by_value.entry(*value).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let root = self.find(self.parent[value]);
            self.parent[value] = root;
        }
        self.parent[value]
    }

    fn union(&mut self, lhs: usize, rhs: usize) {
        let lhs_root = self.find(lhs);
        let rhs_root = self.find(rhs);
        if lhs_root == rhs_root {
            return;
        }

        match self.rank[lhs_root].cmp(&self.rank[rhs_root]) {
            std::cmp::Ordering::Less => self.parent[lhs_root] = rhs_root,
            std::cmp::Ordering::Greater => self.parent[rhs_root] = lhs_root,
            std::cmp::Ordering::Equal => {
                self.parent[rhs_root] = lhs_root;
                self.rank[lhs_root] = self.rank[lhs_root].saturating_add(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_tensorbackend::StorageKind;

    #[test]
    fn plans_diag_diag_partial_as_diag_output() {
        let operands = vec![
            OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
            OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
        ];
        let spec = StructuredContractionSpec {
            input_labels: vec![vec![0, 1], vec![1, 2]],
            output_labels: vec![0, 2],
            retained_labels: Default::default(),
        };

        let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
        assert_eq!(plan.output_axis_classes, vec![0, 0]);
        assert_eq!(plan.output_payload_roots.len(), 1);
    }

    #[test]
    fn plans_general_structured_output_classes() {
        let operands = vec![
            OperandLayout::new(vec![2, 2, 3], vec![0, 0, 1]).unwrap(),
            OperandLayout::new(vec![3, 5, 5], vec![0, 1, 1]).unwrap(),
        ];
        let spec = StructuredContractionSpec {
            input_labels: vec![vec![0, 1, 2], vec![2, 3, 4]],
            output_labels: vec![0, 1, 3, 4],
            retained_labels: Default::default(),
        };

        let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
        assert_eq!(plan.output_axis_classes, vec![0, 0, 1, 1]);
        assert_eq!(plan.output_payload_dims, vec![2, 5]);
    }

    #[test]
    fn detects_payload_repeated_labels() {
        let operands = vec![
            OperandLayout::new(vec![3, 3], vec![0, 1]).unwrap(),
            OperandLayout::new(vec![3, 3], vec![0, 0]).unwrap(),
        ];
        let spec = StructuredContractionSpec {
            input_labels: vec![vec![0, 1], vec![0, 1]],
            output_labels: vec![],
            retained_labels: Default::default(),
        };

        let plan = StructuredContractionPlan::new(&operands, &spec).unwrap();
        assert!(plan.operand_plans[0].has_repeated_roots());
    }

    #[test]
    fn normalizes_repeated_payload_roots_by_extracting_diagonal() {
        let payload =
            tenferro::Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
        let (normalized, roots) = normalize_payload_for_roots(&payload, &[0, 0]).unwrap();

        assert_eq!(normalized.shape(), &[2]);
        assert_eq!(normalized.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
        assert_eq!(roots, vec![0]);
    }

    #[test]
    fn storage_payload_native_roundtrip_preserves_compact_payload() {
        let storage = Storage::new_structured(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![1, 2],
            vec![0, 1, 0],
        )
        .unwrap();

        let native = storage_payload_native(&storage).unwrap();
        assert_eq!(native.shape(), &[2, 3]);
        assert_eq!(
            native.as_slice::<f64>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let rebuilt = storage_from_payload_native(native, &[2, 3], vec![0, 1, 0]).unwrap();
        assert_eq!(rebuilt.storage_kind(), StorageKind::Structured);
        assert_eq!(rebuilt.payload_dims(), &[2, 3]);
        assert_eq!(rebuilt.axis_classes(), &[0, 1, 0]);
        assert_eq!(
            rebuilt.payload_f64_col_major_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }
}
