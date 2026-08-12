use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

#[test]
fn readme_example_uses_public_callback_signature() {
    let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    let sizes = vec![16, 16];

    let (qtci, ranks, errors) = quanticscrossinterpolate_discrete(
        &sizes,
        f,
        None,
        QtciOptions::default().with_tolerance(1e-10),
    )
    .expect("README example should compile and run");

    let value = qtci
        .evaluate(&[4, 9])
        .expect("README example should evaluate");
    assert!((value - 13.0).abs() < 1e-10);
    assert!(!ranks.is_empty());
    assert!(errors.last().copied().unwrap() < 1e-10);
}
