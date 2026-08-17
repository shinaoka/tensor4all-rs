use super::{SweepPlan, TreeAciTraversalStrategy};

fn plan(node_count: usize, edges: &[(usize, usize)]) -> SweepPlan {
    SweepPlan::for_strategy(
        TreeAciTraversalStrategy::MinimumRetracingWalk,
        node_count,
        edges,
        None,
    )
    .unwrap()
}

fn steps(plan: &SweepPlan) -> Vec<(usize, usize, usize)> {
    plan.forward
        .iter()
        .flat_map(|phase| &phase.paths)
        .flat_map(|path| &path.steps)
        .map(|step| (step.edge, step.from, step.to))
        .collect()
}

fn diameter(node_count: usize, edges: &[(usize, usize)]) -> usize {
    let mut adjacency = vec![Vec::new(); node_count];
    for &(left, right) in edges {
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    (0..node_count)
        .flat_map(|start| {
            let adjacency = &adjacency;
            (0..node_count).map(move |end| distance(start, end, adjacency))
        })
        .max()
        .unwrap_or(0)
}

fn distance(start: usize, end: usize, adjacency: &[Vec<usize>]) -> usize {
    let mut stack = vec![(start, usize::MAX, 0usize)];
    while let Some((node, parent, distance)) = stack.pop() {
        if node == end {
            return distance;
        }
        for &neighbor in &adjacency[node] {
            if neighbor != parent {
                stack.push((neighbor, node, distance + 1));
            }
        }
    }
    unreachable!("validated test tree must connect every pair")
}

fn assert_minimum_continuous_walk(node_count: usize, edges: &[(usize, usize)], plan: &SweepPlan) {
    plan.validate(node_count, edges).unwrap();
    let forward = steps(plan);
    assert_eq!(forward.len(), 2 * edges.len() - diameter(node_count, edges));
    assert_eq!(forward.first().map(|step| step.1), Some(plan.start));
    assert!(forward.windows(2).all(|pair| pair[0].2 == pair[1].1));
    let mut visits = vec![0usize; edges.len()];
    for &(edge, _, _) in &forward {
        visits[edge] += 1;
    }
    assert!(visits.iter().all(|visits| *visits >= 1 && *visits <= 2));
    let reverse = plan
        .reverse
        .iter()
        .flat_map(|phase| &phase.paths)
        .flat_map(|path| &path.steps)
        .map(|step| (step.edge, step.from, step.to))
        .collect::<Vec<_>>();
    let expected_reverse = forward
        .iter()
        .rev()
        .map(|&(edge, from, to)| (edge, to, from))
        .collect::<Vec<_>>();
    assert_eq!(reverse, expected_reverse);
}

#[test]
fn path_visits_every_edge_once_like_train_aci() {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
    let plan = plan(5, &edges);
    assert_minimum_continuous_walk(5, &edges, &plan);
    assert_eq!(steps(&plan).len(), edges.len());
}

#[test]
fn stars_binary_and_comb_have_optimal_retracing_length() {
    let cases = [
        (4, vec![(0, 1), (0, 2), (0, 3)]),
        (5, vec![(0, 1), (0, 2), (0, 3), (0, 4)]),
        (7, vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
        (
            8,
            vec![(0, 1), (1, 2), (2, 3), (1, 4), (2, 5), (3, 6), (3, 7)],
        ),
    ];
    for (node_count, edges) in cases {
        assert_minimum_continuous_walk(node_count, &edges, &plan(node_count, &edges));
    }
}

#[test]
fn explicit_root_is_respected_and_ends_at_a_farthest_node() {
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4)];
    let plan = SweepPlan::for_strategy(
        TreeAciTraversalStrategy::MinimumRetracingWalk,
        5,
        &edges,
        Some(1),
    )
    .unwrap();
    let walk = steps(&plan);
    assert_eq!(plan.start, 1);
    assert_eq!(walk.first().map(|step| step.1), Some(1));
    assert_eq!(walk.len(), 2 * edges.len() - 2);
}

#[test]
fn all_labeled_trees_through_seven_nodes_have_minimum_walks() {
    for node_count in 2usize..=7 {
        let sequence_count = node_count.pow((node_count - 2) as u32);
        for encoded in 0..sequence_count {
            let edges = decode_prufer(node_count, encoded);
            assert_minimum_continuous_walk(node_count, &edges, &plan(node_count, &edges));
        }
    }
}

#[test]
fn rejects_non_trees_before_planning() {
    let strategy = TreeAciTraversalStrategy::MinimumRetracingWalk;
    assert!(SweepPlan::for_strategy(strategy, 0, &[], None).is_err());
    assert!(SweepPlan::for_strategy(strategy, 3, &[(0, 1)], None).is_err());
    assert!(SweepPlan::for_strategy(strategy, 3, &[(0, 1), (1, 2), (2, 0)], None).is_err());
    assert!(SweepPlan::for_strategy(strategy, 2, &[(0, 2)], None).is_err());
    assert!(SweepPlan::for_strategy(strategy, 2, &[(0, 0)], None).is_err());
}

#[test]
fn validation_rejects_discontinuity_and_wrong_reverse() {
    let edges = [(0, 1), (1, 2), (1, 3)];
    let mut discontinuous = plan(4, &edges);
    discontinuous.forward[0].paths[0].steps[1].from = 3;
    assert!(discontinuous.validate(4, &edges).is_err());

    let mut invalid_reverse = plan(4, &edges);
    invalid_reverse.reverse[0].paths[0].steps[0].edge = usize::MAX;
    assert!(invalid_reverse.validate(4, &edges).is_err());
}

fn decode_prufer(node_count: usize, mut encoded: usize) -> Vec<(usize, usize)> {
    let mut sequence = Vec::with_capacity(node_count.saturating_sub(2));
    for _ in 0..node_count.saturating_sub(2) {
        sequence.push(encoded % node_count);
        encoded /= node_count;
    }
    let mut degrees = vec![1usize; node_count];
    for &node in &sequence {
        degrees[node] += 1;
    }
    let mut edges = Vec::with_capacity(node_count - 1);
    for node in sequence {
        let leaf = (0..node_count)
            .find(|&candidate| degrees[candidate] == 1)
            .unwrap();
        edges.push((leaf.min(node), leaf.max(node)));
        degrees[leaf] -= 1;
        degrees[node] -= 1;
    }
    let remaining = (0..node_count)
        .filter(|&node| degrees[node] == 1)
        .collect::<Vec<_>>();
    edges.push((
        remaining[0].min(remaining[1]),
        remaining[0].max(remaining[1]),
    ));
    edges.sort_unstable();
    edges
}
