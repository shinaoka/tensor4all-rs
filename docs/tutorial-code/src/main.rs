use tensor4all_simplett::{AbstractTensorTrain, CompressionOptions, SimpleTensorTrain, Tensor3Ops};

fn print_tt_info<T>(label: &str, tt: &SimpleTensorTrain<T>)
where
    T: tensor4all_simplett::TTScalar + std::fmt::Debug,
{
    println!("{label}");
    println!("len = {}", tt.len());
    println!("site_dims = {:?}", tt.site_dims());
    println!("link_dims = {:?}", tt.link_dims());
    println!("rank = {}", tt.rank());

    for (i, core) in tt.site_tensors().iter().enumerate() {
        println!(
            "core {}: left={}, site={}, right={}",
            i,
            core.left_dim(),
            core.site_dim(),
            core.right_dim()
        );
    }

    if !tt.is_empty() {
        let mut indices = vec![0; tt.len()];
        if tt.len() > 1 {
            indices[1] = 1.min(tt.site_dim(1).saturating_sub(1));
        }
        if tt.len() > 2 {
            indices[2] = 2.min(tt.site_dim(2).saturating_sub(1));
        }

        let value = tt.evaluate(&indices).unwrap();
        println!("evaluate{:?} = {:?}", indices, value);
    }

    println!("sum = {:?}", tt.sum());

    let (dense, shape) = tt.full_tensor();
    println!("full_tensor shape = {:?}", shape);
    println!("full_tensor data = {:?}", dense);
    println!();
}

fn main() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
    let tt_value = tt.evaluate(&[0, 1, 2]).unwrap();
    assert!((tt_value - 1.0).abs() < 1e-12);
    let tt_total = tt.sum();
    assert!((tt_total - 24.0).abs() < 1e-12);

    // Compress with a truncation tolerance.
    let options = CompressionOptions {
        tolerance: 1e-10,
        max_bond_dim: Some(20),
        ..Default::default()
    };
    let compressed = tt.compressed(&options).unwrap();
    assert!((compressed.sum() - 24.0).abs() < 1e-10);

    print_tt_info("original tt", &tt);
    print_tt_info("compressed tt", &compressed);
}
