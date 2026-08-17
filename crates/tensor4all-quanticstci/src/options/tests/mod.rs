use super::*;

#[test]
fn test_default_options() {
    let opts = QtciOptions::default();
    assert!((opts.tolerance - 1e-8).abs() < 1e-15);
    assert!(opts.max_bond_dim.is_none());
    assert_eq!(opts.max_iter, 200);
    assert_eq!(opts.n_random_init_pivot, 5);
    assert_eq!(opts.unfolding_scheme, UnfoldingScheme::Interleaved);
}

#[test]
fn test_builder_pattern() {
    let opts = QtciOptions::default()
        .with_tolerance(1e-6)
        .with_max_bond_dim(100)
        .with_maxiter(50);

    assert!((opts.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(opts.max_bond_dim, Some(100));
    assert_eq!(opts.max_iter, 50);
}

#[test]
fn test_to_treetci_options() {
    let opts = QtciOptions::default()
        .with_tolerance(1e-6)
        .with_max_bond_dim(100);

    let tree_opts = opts.to_treetci_options();
    assert!((tree_opts.tolerance - 1e-6).abs() < 1e-15);
    assert_eq!(tree_opts.max_bond_dim, Some(100));
    assert_eq!(tree_opts.max_iter, 200);
    assert!(tree_opts.normalize_error);
}
