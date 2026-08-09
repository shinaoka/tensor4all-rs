// Portions of this file are derived from ITensorNetworks.jl
// (https://github.com/ITensor/ITensorNetworks.jl),
// Copyright 2021 The Simons Foundation, Inc. - All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE in this
// crate). Changes: translated from Julia to Rust and adapted to tensor4all-rs
// tensor and graph types.

use std::fmt::Debug;
use std::hash::Hash;

use num_complex::Complex64;

use crate::node_name_network::NodeNameNetwork;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TdvpRegionKind {
    OneSite,
    TwoSite,
    SiteCorrection,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TdvpRegionStep<V> {
    pub(crate) nodes: Vec<V>,
    pub(crate) new_center: V,
    pub(crate) exponent_step: Complex64,
    pub(crate) kind: TdvpRegionKind,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TdvpRegionPlan<V> {
    pub(crate) steps: Vec<TdvpRegionStep<V>>,
    pub(crate) nsite: usize,
    pub(crate) order: usize,
}

impl<V> TdvpRegionPlan<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    pub(crate) fn new(
        network: &NodeNameNetwork<V>,
        root: &V,
        nsite: usize,
        order: usize,
        exponent_step: Complex64,
    ) -> Option<Self> {
        if nsite != 1 && nsite != 2 {
            return None;
        }
        let weights = applyexp_sub_steps(order)?;
        let base_sweep = first_order_sweep(network, root, nsite, exponent_step)?;
        let mut steps = Vec::new();
        for (substep, weight) in weights.iter().enumerate() {
            let mut region_plan = base_sweep
                .iter()
                .map(|step| {
                    let mut step = step.clone();
                    step.exponent_step *= *weight;
                    step
                })
                .collect::<Vec<_>>();
            if (substep + 1) % 2 == 0 {
                region_plan = reverse_regions(&region_plan)?;
            }
            steps.extend(region_plan);
        }
        Some(Self {
            steps,
            nsite,
            order,
        })
    }
}

pub(crate) fn applyexp_sub_steps(order: usize) -> Option<Vec<f64>> {
    match order {
        1 => Some(vec![1.0]),
        2 => Some(vec![0.5, 0.5]),
        4 => {
            let s = (2.0 - 2.0_f64.powf(1.0 / 3.0)).recip();
            Some(vec![s / 2.0, s / 2.0, 0.5 - s, 0.5 - s, s / 2.0, s / 2.0])
        }
        _ => None,
    }
}

fn first_order_sweep<V>(
    network: &NodeNameNetwork<V>,
    root: &V,
    nsite: usize,
    exponent_step: Complex64,
) -> Option<Vec<TdvpRegionStep<V>>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    match nsite {
        1 => {
            let vertices = network.post_order_dfs(root)?;
            Some(
                vertices
                    .into_iter()
                    .map(|node| TdvpRegionStep {
                        nodes: vec![node.clone()],
                        new_center: node,
                        exponent_step,
                        kind: TdvpRegionKind::OneSite,
                    })
                    .collect(),
            )
        }
        2 => {
            // Root-edge-first outward walk (pre-order DFS edges). Starting each
            // half-sweep at the edge containing the sweep root keeps every gauge
            // move inside already-evolved regions, which is required for the
            // symmetric projector-splitting integrator; the old post-order
            // (root-edge-last) ordering transported the gauge across unevolved
            // edges at the start and turning point of each half-sweep. Matches
            // ITensorNetworks.jl at NamedGraphs >= 0.11 (see benchmarks/results/
            // 2026-07-22-treetn-tdvp-itensornetworks-1t.md).
            let edges = pre_order_dfs_edges_by_name(network, root)?;
            let mut steps = Vec::new();
            let last_edge = edges.len().saturating_sub(1);
            for (j, (parent, child)) in edges.iter().enumerate() {
                // The center after a two-site step sits on the vertex shared
                // with the next region so the following site correction acts on
                // the overlap; the final edge parks the center on its far side.
                let center = match edges.get(j + 1) {
                    Some((next_parent, next_child)) => {
                        if parent == next_parent || parent == next_child {
                            parent.clone()
                        } else {
                            child.clone()
                        }
                    }
                    None => child.clone(),
                };
                let other = if center == *parent {
                    child.clone()
                } else {
                    parent.clone()
                };
                steps.push(TdvpRegionStep {
                    nodes: vec![other, center.clone()],
                    new_center: center.clone(),
                    exponent_step,
                    kind: TdvpRegionKind::TwoSite,
                });
                if j < last_edge {
                    steps.push(TdvpRegionStep {
                        nodes: vec![center.clone()],
                        new_center: center,
                        exponent_step: -exponent_step,
                        kind: TdvpRegionKind::SiteCorrection,
                    });
                }
            }
            Some(steps)
        }
        _ => None,
    }
}

fn reverse_regions<V>(steps: &[TdvpRegionStep<V>]) -> Option<Vec<TdvpRegionStep<V>>>
where
    V: Clone,
{
    let mut reversed = Vec::with_capacity(steps.len());
    for step in steps.iter().rev() {
        let mut nodes = step.nodes.clone();
        nodes.reverse();
        let new_center = nodes.last()?.clone();
        reversed.push(TdvpRegionStep {
            nodes,
            new_center,
            exponent_step: step.exponent_step,
            kind: step.kind,
        });
    }
    Some(reversed)
}

/// Tree edges as `(parent, child)` pairs in a parents-before-children order
/// (reverse post-order), so the edge containing `root` comes first and each
/// subsequent edge attaches to an already-visited vertex.
fn pre_order_dfs_edges_by_name<V>(network: &NodeNameNetwork<V>, root: &V) -> Option<Vec<(V, V)>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let root_idx = network.node_index(root)?;
    let post_order = network.post_order_dfs_by_index(root_idx);
    let mut edges = Vec::new();
    for &child in post_order.iter().rev() {
        if child == root_idx {
            continue;
        }
        let path = network.path_between(child, root_idx)?;
        let parent = *path.get(1)?;
        let child_name = network.node_name(child)?.clone();
        let parent_name = network.node_name(parent)?.clone();
        edges.push((parent_name, child_name));
    }
    Some(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_abc() -> NodeNameNetwork<&'static str> {
        let mut network = NodeNameNetwork::new();
        network.add_node("A").unwrap();
        network.add_node("B").unwrap();
        network.add_node("C").unwrap();
        network.add_edge(&"A", &"B").unwrap();
        network.add_edge(&"B", &"C").unwrap();
        network
    }

    fn star_abcd() -> NodeNameNetwork<&'static str> {
        let mut network = NodeNameNetwork::new();
        for node in ["A", "B", "C", "D"] {
            network.add_node(node).unwrap();
        }
        network.add_edge(&"A", &"B").unwrap();
        network.add_edge(&"A", &"C").unwrap();
        network.add_edge(&"A", &"D").unwrap();
        network
    }

    #[test]
    fn reverse_regions_rejects_empty_nodes() {
        let step = TdvpRegionStep {
            nodes: Vec::<&str>::new(),
            new_center: "unused",
            exponent_step: Complex64::new(0.0, -0.2),
            kind: TdvpRegionKind::OneSite,
        };

        assert!(reverse_regions(&[step]).is_none());
    }

    #[test]
    fn applyexp_sub_steps_match_itensornetworks_orders() {
        assert_eq!(applyexp_sub_steps(1).unwrap(), vec![1.0]);
        assert_eq!(applyexp_sub_steps(2).unwrap(), vec![0.5, 0.5]);

        let fourth = applyexp_sub_steps(4).unwrap();
        assert_eq!(fourth.len(), 6);
        let total: f64 = fourth.iter().sum();
        assert!((total - 1.0).abs() < 1.0e-14);
        assert!(applyexp_sub_steps(3).is_none());
    }

    #[test]
    fn two_site_plan_starts_at_root_edge_with_corrections_except_last_edge() {
        let network = chain_abc();
        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"B", 2, 1, exponent).unwrap();

        let regions: Vec<_> = plan
            .steps
            .iter()
            .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
            .collect();
        // The first region contains the sweep root B; the walk moves outward.
        assert_eq!(
            regions,
            vec![
                (TdvpRegionKind::TwoSite, vec!["C", "B"], exponent),
                (TdvpRegionKind::SiteCorrection, vec!["B"], -exponent),
                (TdvpRegionKind::TwoSite, vec!["B", "A"], exponent),
            ]
        );
    }

    #[test]
    fn second_order_plan_reverses_even_substep_regions() {
        let network = chain_abc();
        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"B", 2, 2, exponent).unwrap();

        let regions: Vec<_> = plan
            .steps
            .iter()
            .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
            .collect();
        let half = exponent * 0.5;
        assert_eq!(
            regions,
            vec![
                (TdvpRegionKind::TwoSite, vec!["C", "B"], half),
                (TdvpRegionKind::SiteCorrection, vec!["B"], -half),
                (TdvpRegionKind::TwoSite, vec!["B", "A"], half),
                (TdvpRegionKind::TwoSite, vec!["A", "B"], half),
                (TdvpRegionKind::SiteCorrection, vec!["B"], -half),
                (TdvpRegionKind::TwoSite, vec!["B", "C"], half),
            ]
        );
    }

    #[test]
    fn two_site_star_plan_walks_outward_from_root_leaf() {
        // Regression test for the root-edge-first ordering (issue #560): on a
        // branching tree the half-sweep must start at the edge containing the
        // sweep root and keep every subsequent region attached to an
        // already-visited vertex, with the site correction on the overlap.
        let network = star_abcd();
        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"B", 2, 1, exponent).unwrap();

        let regions: Vec<_> = plan
            .steps
            .iter()
            .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
            .collect();
        assert_eq!(
            regions,
            vec![
                (TdvpRegionKind::TwoSite, vec!["B", "A"], exponent),
                (TdvpRegionKind::SiteCorrection, vec!["A"], -exponent),
                (TdvpRegionKind::TwoSite, vec!["D", "A"], exponent),
                (TdvpRegionKind::SiteCorrection, vec!["A"], -exponent),
                (TdvpRegionKind::TwoSite, vec!["A", "C"], exponent),
            ]
        );
    }

    #[test]
    fn two_site_plan_keeps_every_region_attached_to_visited_vertices() {
        // Two branches of depth 2 force a jump between subtrees; even then the
        // walk must stay contiguous: the first region contains the sweep root
        // and every later region shares a vertex with the already-visited set.
        let mut network = NodeNameNetwork::new();
        for node in ["A", "B", "C", "D", "E"] {
            network.add_node(node).unwrap();
        }
        network.add_edge(&"A", &"B").unwrap();
        network.add_edge(&"B", &"C").unwrap();
        network.add_edge(&"A", &"D").unwrap();
        network.add_edge(&"D", &"E").unwrap();

        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"A", 2, 1, exponent).unwrap();

        let two_site_regions: Vec<Vec<&str>> = plan
            .steps
            .iter()
            .filter(|step| step.kind == TdvpRegionKind::TwoSite)
            .map(|step| step.nodes.clone())
            .collect();
        assert_eq!(two_site_regions.len(), 4);
        assert!(two_site_regions[0].contains(&"A"));
        let mut visited: std::collections::HashSet<&str> =
            two_site_regions[0].iter().copied().collect();
        for region in &two_site_regions[1..] {
            assert!(
                region.iter().any(|node| visited.contains(node)),
                "region {region:?} is detached from visited set {visited:?}"
            );
            visited.extend(region.iter().copied());
        }
    }

    #[test]
    fn one_site_plan_uses_post_order_vertices() {
        let network = star_abcd();
        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"A", 1, 1, exponent).unwrap();

        assert_eq!(plan.nsite, 1);
        assert_eq!(plan.order, 1);
        assert_eq!(
            plan.steps
                .iter()
                .map(|step| (step.kind, step.nodes.clone()))
                .collect::<Vec<_>>(),
            vec![
                (TdvpRegionKind::OneSite, vec!["B"]),
                (TdvpRegionKind::OneSite, vec!["C"]),
                (TdvpRegionKind::OneSite, vec!["D"]),
                (TdvpRegionKind::OneSite, vec!["A"]),
            ]
        );
    }
}
