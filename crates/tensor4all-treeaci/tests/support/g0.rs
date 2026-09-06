//! Exact separable inputs for the low-temperature Green's function in #741.

use std::f64::consts::PI;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

pub(crate) type TestResult<T> = Result<T, Box<dyn std::error::Error>>;
type Fixture = (Vec<DynIndex>, Vec<TreeTN<IdxTensor, usize>>);

pub(crate) fn fixture(r: usize, mode: &str) -> TestResult<Fixture> {
    assert!((2..=10).contains(&r));
    let sites: Vec<_> = (0..3 * r).map(|_| DynIndex::new_dyn(2)).collect();
    let edges = match mode {
        "cttn" | "swap" => {
            let mut edges = if mode == "swap" {
                vec![(0, 2), (1, 2)]
            } else {
                vec![(0, 1), (1, 2)]
            };
            for var in 0..3 {
                edges.extend((1..r).map(|bit| (var + 3 * (bit - 1), var + 3 * bit)));
            }
            edges
        }
        "nblock" => {
            let order: Vec<_> = (0..r)
                .map(|bit| 2 + 3 * bit)
                .chain((0..r).flat_map(|bit| [3 * bit, 1 + 3 * bit]))
                .collect();
            order.windows(2).map(|pair| (pair[0], pair[1])).collect()
        }
        _ => return Err("topology must be cttn, swap, or nblock".into()),
    };
    let mut adjacency = vec![Vec::new(); sites.len()];
    for &(a, b) in &edges {
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    let mut inputs = Vec::new();
    for var in 0..4 {
        // The unique path between the first and last owner carries a rank-2
        // rotation/affine state; intervening non-owner sites pass it through.
        let mut path = Vec::new();
        if var < 3 {
            let mut parents = vec![usize::MAX; sites.len()];
            parents[var] = var;
            let mut stack = vec![var];
            while let Some(node) = stack.pop() {
                for &next in &adjacency[node] {
                    if parents[next] == usize::MAX {
                        parents[next] = node;
                        stack.push(next);
                    }
                }
            }
            let mut node = var + 3 * (r - 1);
            while node != var {
                path.push(node);
                node = parents[node];
            }
            path.push(var);
            path.reverse();
            assert_eq!(
                path.iter()
                    .copied()
                    .filter(|node| node % 3 == var)
                    .collect::<Vec<_>>(),
                (0..r).map(|bit| var + 3 * bit).collect::<Vec<_>>()
            );
        }
        let bonds: Vec<_> = edges
            .iter()
            .map(|&(a, b)| {
                DynIndex::new_dyn(
                    if path.windows(2).any(|pair| pair == [a, b] || pair == [b, a]) {
                        2
                    } else {
                        1
                    },
                )
            })
            .collect();
        let mut tensors = Vec::new();
        for (node, site) in sites.iter().enumerate() {
            let mut indices = vec![site.clone()];
            let mut prev = None;
            let mut next = None;
            let position = path.iter().position(|&site| site == node);
            for (edge, &(a, b)) in edges.iter().enumerate() {
                if a == node || b == node {
                    let other = if a == node { b } else { a };
                    if bonds[edge].dim() == 2 {
                        if position.is_some_and(|pos| pos > 0 && path[pos - 1] == other) {
                            prev = Some(indices.len());
                        } else {
                            next = Some(indices.len());
                        }
                    }
                    indices.push(bonds[edge].clone());
                }
            }
            let dims: Vec<_> = indices.iter().map(IndexLike::dim).collect();
            let values = (0..dims.iter().product())
                .map(|flat| {
                    if var == 3 {
                        return Complex64::new(0.0, 0.0);
                    }
                    if position.is_none() {
                        return Complex64::new(1.0, 0.0);
                    }
                    let mut rest = flat;
                    let coords: Vec<_> = dims
                        .iter()
                        .map(|d| {
                            let x = rest % d;
                            rest /= d;
                            x
                        })
                        .collect();
                    let bit = node / 3;
                    let m = if node % 3 != var {
                        [1.0, 0.0, 0.0, 1.0]
                    } else if var < 2 {
                        let (s, c) = (coords[0] as f64 * PI / (1usize << bit) as f64).sin_cos();
                        [c, -s, s, c]
                    } else {
                        [
                            1.0,
                            coords[0] as f64 * 2.0 * PI * 0.01 * (1usize << (r - 1 - bit)) as f64,
                            0.0,
                            1.0,
                        ]
                    };
                    let initial = if var < 2 { [1.0, 0.0] } else { [0.0, 1.0] };
                    let incoming = prev.map_or(initial, |axis| {
                        if coords[axis] == 0 {
                            [1.0, 0.0]
                        } else {
                            [0.0, 1.0]
                        }
                    });
                    let after = [
                        m[0] * incoming[0] + m[1] * incoming[1],
                        m[2] * incoming[0] + m[3] * incoming[1],
                    ];
                    let value = next.map_or_else(
                        || {
                            let beta = if var < 2 {
                                0.0
                            } else {
                                -2.0 * PI * 0.01 * (1usize << (r - 1)) as f64 + PI * 0.01
                            };
                            after[0] + beta * after[1]
                        },
                        |axis| after[coords[axis]],
                    );
                    Complex64::new(value, 0.0)
                })
                .collect();
            tensors.push(IdxTensor::from_dense(indices, values)?);
        }
        inputs.push(TreeTN::from_tensors(tensors, (0..3 * r).collect())?);
    }
    Ok((sites, inputs))
}

pub(crate) fn operator(
    batch: tensor4all_treeaci::TreeElementwiseBatch<'_, Complex64>,
    output: &mut [Complex64],
) -> tensor4all_treeaci::Result<()> {
    for (p, value) in output.iter_mut().enumerate() {
        *value = 1.0
            / (Complex64::new(0.5, 0.0)
                + 2.0 * batch.get(0, p)?
                + 2.0 * batch.get(1, p)?
                + Complex64::i() * batch.get(2, p)?
                - batch.get(3, p)?);
    }
    Ok(())
}

pub(crate) fn exact(r: usize, x: usize, y: usize, n: usize) -> Complex64 {
    let size = (1usize << r) as f64;
    1.0 / Complex64::new(
        0.5 + 2.0 * (2.0 * PI * x as f64 / size).cos() + 2.0 * (2.0 * PI * y as f64 / size).cos(),
        2.0 * PI * 0.01 * (n as f64 - size / 2.0 + 0.5),
    )
}

pub(crate) fn options() -> tensor4all_treeaci::TreeAciOptions<usize> {
    tensor4all_treeaci::TreeAciOptions {
        tolerance: 3.183055503898938e-3,
        scale_tolerance: false,
        max_sweeps: 50,
        max_bond_dim: Some(2000),
        rng_seed: 42,
        max_working_bytes: 8usize << 30,
        max_frame_bytes: 8usize << 30,
        max_sample_arena_bytes: 8usize << 30,
        message_cache_max_bytes: 8usize << 30,
        ..Default::default()
    }
}
