//! Regression test: QR factorize with non-pivoting QR used to truncate rank
//! incorrectly by checking only R's diagonal elements (which can be zero even
//! when the row has significant off-diagonal entries).
//!
//! A rank-4 tensor [5,2,2,5] with specific complex data gave QR reconstruction
//! error ~4e-3 when factorized with left_inds = first 2 indices (10x10 matrix).
//! SVD gives ~1e-14 on the same data.

use num_complex::Complex64;
use tensor4all_core::{
    factorize, svd, DynIndex, FactorizeOptions, TensorContractionLike, TensorDynLen,
};

/// Create a [5,2,2,5] tensor with data that triggers the QR bug.
fn make_buggy_tensor() -> TensorDynLen {
    let idx_a = DynIndex::new_dyn_with_tag(5, "a").unwrap();
    let idx_b = DynIndex::new_dyn_with_tag(2, "b").unwrap();
    let idx_c = DynIndex::new_dyn_with_tag(2, "c").unwrap();
    let idx_d = DynIndex::new_dyn_with_tag(5, "d").unwrap();

    #[rustfmt::skip]
    #[allow(clippy::excessive_precision)]
    let data: Vec<Complex64> = vec![
        Complex64::new(-3.71572844459456819e-1, 1.10576880726913718e0),
        Complex64::new(-4.04336097845196418e-1, 9.74397622910009309e-1),
        Complex64::new(-4.34288047537135569e-1, 8.40845498967369598e-1),
        Complex64::new(-4.60776040616710292e-1, 7.05852518714040977e-1),
        Complex64::new(-4.83177747626307552e-1, 5.70139068997173926e-1),
        Complex64::new(-6.11349208859297444e-1, 4.23958514197407321e-1),
        Complex64::new(-5.58197149046672059e-1, 3.27467452276840265e-1),
        Complex64::new(-5.03412000507335811e-1, 2.32621263581571952e-1),
        Complex64::new(-4.46936107821081174e-1, 1.40151864381165892e-1),
        Complex64::new(-3.88712353611773698e-1, 5.07443970399309510e-2),
        Complex64::new(-5.00909132547974756e-1, 4.34401720363751020e-1),
        Complex64::new(-5.13432052585185095e-1, 2.99309497062528462e-1),
        Complex64::new(-5.20261403170824366e-1, 1.65500558370423573e-1),
        Complex64::new(-5.20971728152462066e-1, 3.35793079165228534e-2),
        Complex64::new(-5.15203220470327072e-1, -9.58860571866677891e-2),
        Complex64::new(-3.28687772890678187e-1, -3.49684461485573472e-2),
        Complex64::new(-2.66817191111389795e-1, -1.16411468010992350e-1),
        Complex64::new(-2.03066834357909759e-1, -1.93071336272502281e-1),
        Complex64::new(-1.37417860123768942e-1, -2.64500241148672544e-1),
        Complex64::new(-6.98697577612843179e-2, -3.30318811412943447e-1),
        Complex64::new(-8.11349898261427560e-1, -4.33198612198789634e-1),
        Complex64::new(-6.71085248173132531e-1, -4.78459288017937967e-1),
        Complex64::new(-5.30296856264999117e-1, -5.17864445594039546e-1),
        Complex64::new(-3.89608080661461853e-1, -5.50804722736728536e-1),
        Complex64::new(-2.49614843933616520e-1, -5.76741673317427872e-1),
        Complex64::new(-9.80947847932863382e-1, -1.25854796408197678e0),
        Complex64::new(-7.69641501907134207e-1, -1.25244965717400536e0),
        Complex64::new(-5.58314422435943869e-1, -1.23686746041821261e0),
        Complex64::new(-3.48211449587025823e-1, -1.21136282863380429e0),
        Complex64::new(-1.40531873429276533e-1, -1.17558442711853206e0),
        Complex64::new(-1.10883645204220255e-1, -5.95214172483509718e-1),
        Complex64::new(2.60500236805106984e-2, -6.05844013321419905e-1),
        Complex64::new(1.60682862054805475e-1, -6.08340634202959696e-1),
        Complex64::new(2.92544552990225359e-1, -6.02504924936180464e-1),
        Complex64::new(4.21198177336995327e-1, -5.88232069267410451e-1),
        Complex64::new(6.35781407299406409e-2, -1.12927461108221205e0),
        Complex64::new(2.63032454695865336e-1, -1.07227501285869997e0),
        Complex64::new(4.56811140645534908e-1, -1.00453116631252715e0),
        Complex64::new(6.43965906382332975e-1, -9.26096107336321639e-1),
        Complex64::new(8.23624731720701164e-1, -8.37132899492684879e-1),
        Complex64::new(-1.11655928808291649e0, -1.84371890605578281e0),
        Complex64::new(-8.65831012572003678e-1, -1.80105809523515514e0),
        Complex64::new(-6.14678725631454315e-1, -1.74679507265485223e0),
        Complex64::new(-3.64766958064558056e-1, -1.68064467072662316e0),
        Complex64::new(-1.17711687871958195e-1, -1.60241323426590943e0),
        Complex64::new(-1.19792081070753320e0, -2.03090463258129850e0),
        Complex64::new(-9.49636706526581609e-1, -1.97616707986736140e0),
        Complex64::new(-6.99651476143251960e-1, -1.90987555773727924e0),
        Complex64::new(-4.49742207913458369e-1, -1.83183453806120777e0),
        Complex64::new(-2.01650618764212669e-1, -1.74192997107147596e0),
        Complex64::new(1.24931490646232823e-1, -1.51200474130063967e0),
        Complex64::new(3.61678438489079102e-1, -1.40942606139027693e0),
        Complex64::new(5.91126695532169566e-1, -1.29479127785836989e0),
        Complex64::new(8.11964776490311846e-1, -1.16832500924784055e0),
        Complex64::new(1.02298046948695220e0, -1.03036467495848671e0),
        Complex64::new(4.29306490375584229e-2, -1.64013480279027046e0),
        Complex64::new(2.82371089365989703e-1, -1.52651377771168550e0),
        Complex64::new(5.15114640322993145e-1, -1.40122745655032332e0),
        Complex64::new(7.39691135463315441e-1, -1.26453538606708071e0),
        Complex64::new(9.54726803804743396e-1, -1.11679836591468007e0),
        Complex64::new(-1.19148164538101109e0, -1.75326140090760063e0),
        Complex64::new(-9.89870867606832405e-1, -1.71292465872752420e0),
        Complex64::new(-7.84772951843899058e-1, -1.66345655088099487e0),
        Complex64::new(-5.77728091554425838e-1, -1.60467295157346812e0),
        Complex64::new(-3.70267063406226449e-1, -1.53644703137484107e0),
        Complex64::new(-1.06027083447361004e0, -1.05382227219306768e0),
        Complex64::new(-9.43104662140722638e-1, -1.04828754357645804e0),
        Complex64::new(-8.20758186548817714e-1, -1.03809998965632411e0),
        Complex64::new(-6.94218108369098807e-1, -1.02302859735851270e0),
        Complex64::new(-5.64493402384733667e-1, -1.00286426857081334e0),
        Complex64::new(-1.63898567122735345e-1, -1.45871402248950677e0),
        Complex64::new(3.99030855813030383e-2, -1.37147552204913481e0),
        Complex64::new(2.39709823505540237e-1, -1.27480327252205683e0),
        Complex64::new(4.34151105583521046e-1, -1.16884236287381738e0),
        Complex64::new(6.21924549404907712e-1, -1.05381379930190144e0),
        Complex64::new(-4.32606486959338732e-1, -9.77423671126390725e-1),
        Complex64::new(-2.99584349188268984e-1, -9.46552955385228545e-1),
        Complex64::new(-1.66449707840045258e-1, -9.10131290046294672e-1),
        Complex64::new(-3.42122954081100009e-2, -8.68074172073789896e-1),
        Complex64::new(9.61396609697112298e-2, -8.20336467365984312e-1),
        Complex64::new(-7.77623072461917197e-1, -7.79856532220169624e-2),
        Complex64::new(-7.68372531234836731e-1, -1.15306617523929331e-1),
        Complex64::new(-7.53012645275687009e-1, -1.53709238037068874e-1),
        Complex64::new(-7.31763058006248635e-1, -1.92904077720730344e-1),
        Complex64::new(-7.04894256720658863e-1, -2.32582381247093684e-1),
        Complex64::new(-3.40835894071453149e-1, 9.58949033139879781e-1),
        Complex64::new(-4.43557614368768061e-1, 8.85850275870377124e-1),
        Complex64::new(-5.40550401347390963e-1, 8.05801091943465497e-1),
        Complex64::new(-6.31205201328519538e-1, 7.19103044500315036e-1),
        Complex64::new(-7.14980877061578091e-1, 6.26117260804430531e-1),
        Complex64::new(-6.72724585056755031e-1, -2.72418754102105920e-1),
        Complex64::new(-6.35616817902031173e-1, -3.12074065777999621e-1),
        Complex64::new(-5.93974334153578032e-1, -3.51198549754291822e-1),
        Complex64::new(-5.48236926563769544e-1, -3.89435070706681152e-1),
        Complex64::new(-4.98876291321477816e-1, -4.26422527413062913e-1),
        Complex64::new(-7.91407828506232702e-1, 5.27263298108555101e-1),
        Complex64::new(-8.60090885107532732e-1, 4.23017447235324306e-1),
        Complex64::new(-9.20711449481505073e-1, 3.13910478597508202e-1),
        Complex64::new(-9.73028879931346569e-1, 2.00524840910069502e-1),
        Complex64::new(-1.01688110680129151e0, 8.34913283284349772e-2),
    ];

    TensorDynLen::from_dense(vec![idx_a, idx_b, idx_c, idx_d], data).unwrap()
}

fn reconstruction_error(t: &TensorDynLen, left_inds: &[DynIndex], opts: &FactorizeOptions) -> f64 {
    let result = factorize(t, left_inds, opts).unwrap();
    let recon = result.left.contract_pair(&result.right).unwrap();
    let neg = recon
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = t.add(&neg).unwrap();
    diff.norm().unwrap()
}

fn svd_reconstruction_error(t: &TensorDynLen, left_inds: &[DynIndex]) -> f64 {
    let (u, s, v) = svd::<Complex64>(t, left_inds).unwrap();
    let mut perm = vec![v.indices.len() - 1];
    perm.extend(0..v.indices.len() - 1);
    let vh = v.conj().permute(&perm).unwrap();
    let svh = s
        .contract_pair(&vh)
        .unwrap()
        .replaceind(&s.indices[1], &u.indices[u.indices.len() - 1])
        .unwrap();
    let recon = u.contract_pair(&svh).unwrap();
    let neg = recon
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = t.add(&neg).unwrap();
    diff.norm().unwrap()
}

/// Regression: QR and SVD should both reconstruct this tensor to machine precision.
#[test]
fn test_qr_reconstruction_regression() {
    let t = make_buggy_tensor(); // [a(5), b(2), c(2), d(5)]
    let a = t.indices[0].clone();
    let b = t.indices[1].clone();
    let left_inds = [a, b];

    let matrix_i = DynIndex::new_dyn_with_tag(10, "row").unwrap();
    let matrix_j = DynIndex::new_dyn_with_tag(10, "col").unwrap();
    let matrix = TensorDynLen::from_dense(
        vec![matrix_i.clone(), matrix_j],
        t.to_vec::<Complex64>().unwrap(),
    )
    .unwrap();

    let matrix_svd_err = svd_reconstruction_error(&matrix, &[matrix_i]);
    let direct_svd_err = svd_reconstruction_error(&t, &left_inds);
    let qr_err = reconstruction_error(&t, &left_inds, &FactorizeOptions::qr());
    let svd_err = reconstruction_error(&t, &left_inds, &FactorizeOptions::svd());

    assert!(
        matrix_svd_err < 1e-10,
        "matrix-level direct SVD should be exact, got err={matrix_svd_err:.3e}"
    );
    assert!(
        direct_svd_err < 1e-10,
        "direct SVD should be exact, got err={direct_svd_err:.3e}"
    );
    assert!(
        svd_err < 1e-10,
        "factorize(SVD) should be exact, got err={svd_err:.3e}"
    );
    assert!(
        qr_err < 1e-10,
        "QR should be exact (like SVD), got err={qr_err:.3e} (factorize(SVD)={svd_err:.3e})"
    );
}
