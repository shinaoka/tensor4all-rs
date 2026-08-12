use super::{
    union_with_history, DefaultProposer, PivotCandidateProposer, SimpleProposer,
    TruncatedDefaultProposer,
};
use crate::{AllEdges, EdgeVisitor, SubtreeKey, TreeTCI2, TreeTciEdge, TreeTciGraph};
use std::collections::HashMap;
use tensor4all_core::ColMajorArray;

fn sample_graph() -> TreeTciGraph {
    TreeTciGraph::new(
        7,
        &[
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(4, 5),
            TreeTciEdge::new(4, 6),
        ],
    )
    .unwrap()
}

#[test]
fn all_edges_visits_all_edges_in_sorted_order() {
    let tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    assert_eq!(
        AllEdges.visit_order(&tci),
        vec![
            TreeTciEdge::new(0, 1),
            TreeTciEdge::new(1, 2),
            TreeTciEdge::new(1, 3),
            TreeTciEdge::new(3, 4),
            TreeTciEdge::new(4, 5),
            TreeTciEdge::new(4, 6),
        ]
    );
}

#[test]
fn default_proposer_matches_neighbor_product_assembly() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();

    let (iset, jset) = DefaultProposer
        .candidates(&tci, TreeTciEdge::new(1, 3))
        .unwrap();

    assert_eq!(
        iset,
        vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![0, 1, 1],
            vec![1, 0, 0],
            vec![1, 1, 0],
            vec![1, 0, 1],
            vec![1, 1, 1],
        ]
    );
    assert_eq!(
        jset,
        vec![
            vec![0, 0, 0, 0],
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 1],
            vec![1, 1, 0, 1],
        ]
    );
}

#[test]
fn default_proposer_unions_history_candidates() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();
    // History entry: ColMajorArray shape [4, 1], single column [1, 1, 1, 1]
    tci.ijset_history.push(HashMap::from([(
        SubtreeKey::new(vec![0, 1, 2, 3]),
        ColMajorArray::new(vec![1, 1, 1, 1], vec![4, 1]).unwrap(),
    )]));

    let (iset, _jset) = DefaultProposer
        .candidates(&tci, TreeTciEdge::new(3, 4))
        .unwrap();

    assert!(iset.contains(&vec![1, 1, 1, 1]));
}

#[test]
fn union_with_history_dedups_in_first_occurrence_order() {
    let key = SubtreeKey::new(vec![0, 1, 2]);
    let history = HashMap::from([(
        key.clone(),
        ColMajorArray::new(vec![1, 1, 1, 0, 1, 2], vec![3, 2]).unwrap(), // [1,1,1], [0,1,2]
    )]);

    // `values` contains duplicates (the cartesian pivot expansion can produce
    // them); history adds two columns, one of which is already present.
    let values = vec![vec![0, 0, 0], vec![0, 0, 0], vec![1, 1, 1], vec![2, 2, 2]];
    let out = union_with_history(values, Some(&history), &key).unwrap();

    // First occurrence wins, both within `values` and across history: the
    // already-seen [1,1,1] is not appended again, [0,1,2] is new and appended.
    assert_eq!(
        out,
        vec![vec![0, 0, 0], vec![1, 1, 1], vec![2, 2, 2], vec![0, 1, 2]]
    );
}

#[test]
fn simple_proposer_is_deterministic_for_a_fixed_seed() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();

    let proposer = SimpleProposer::seeded(7);
    let first = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();
    let second = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();

    assert_eq!(first, second);
    assert!(!first.0.is_empty());
    assert!(!first.1.is_empty());
    assert!(first.0.iter().all(|candidate| candidate.len() == 3));
    assert!(first.1.iter().all(|candidate| candidate.len() == 4));
}

#[test]
fn truncated_default_proposer_truncates_default_candidates_in_order() {
    let mut tci = TreeTCI2::<f64>::new(vec![2; 7], sample_graph()).unwrap();
    tci.add_global_pivots(&[vec![0, 0, 0, 0, 0, 0, 0], vec![1, 0, 1, 0, 1, 0, 1]])
        .unwrap();

    let default_candidates = DefaultProposer
        .candidates(&tci, TreeTciEdge::new(1, 3))
        .unwrap();
    let proposer = TruncatedDefaultProposer::seeded(7);
    let first = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();
    let second = proposer.candidates(&tci, TreeTciEdge::new(1, 3)).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.0.len(), 4);
    assert_eq!(first.1.len(), 4);
    assert_eq!(first.1, default_candidates.1);
    assert!(first
        .0
        .iter()
        .all(|candidate| default_candidates.0.contains(candidate)));

    let default_positions = first
        .0
        .iter()
        .map(|candidate| {
            default_candidates
                .0
                .iter()
                .position(|value| value == candidate)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(default_positions
        .windows(2)
        .all(|window| window[0] < window[1]));
}
