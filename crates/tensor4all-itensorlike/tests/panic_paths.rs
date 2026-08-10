use std::error::Error;
use tensor4all_core::{DynIndex, TensorDynLen, TensorDynLenError};
use tensor4all_itensorlike::{TensorTrain, TensorTrainError};
use tensor4all_treetn::TreeTN;

fn site_tensor(size: usize, data: Vec<f64>) -> TensorDynLen {
    let site = DynIndex::new_dyn(size);
    TensorDynLen::from_dense(vec![site], data).unwrap()
}

#[test]
fn tensor_accessors_report_invalid_sites() {
    let mut tt = TensorTrain::default();

    let err = tt.tensor(0).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 0, length: 0 }
    ));

    let err = tt.tensor_mut(0).unwrap_err();
    assert!(matches!(
        err,
        TensorTrainError::SiteOutOfBounds { site: 0, length: 0 }
    ));
}

#[test]
fn inner_reports_length_mismatch() {
    let left = TensorTrain::new(vec![site_tensor(2, vec![1.0, 2.0])]).unwrap();
    let right = TensorTrain::default();

    let err = left.inner(&right).unwrap_err();
    assert!(err.to_string().contains("same length"));
}

#[test]
fn to_dense_preserves_contraction_error_source() {
    let mut tree = TreeTN::<TensorDynLen, usize>::new();
    tree.add_tensor(0, site_tensor(2, vec![1.0, 2.0])).unwrap();
    tree.add_tensor(1, site_tensor(2, vec![3.0, 4.0])).unwrap();
    let tt = TensorTrain::from_treetn(tree).unwrap();

    let error = tt.to_dense().unwrap_err();
    assert!(matches!(
        error,
        TensorTrainError::OperationErrorSource { .. }
    ));
    assert!(error.source().is_some());
}

#[test]
fn dense_maxabs_preserves_typed_tensor_error_source() {
    let tt = TensorTrain::new(vec![TensorDynLen::scalar(f64::NAN).unwrap()]).unwrap();
    let error = tt.dense_maxabs().unwrap_err();

    assert!(matches!(
        error,
        TensorTrainError::TensorDynLen {
            source: TensorDynLenError::NaNInput {
                operation: "maxabs"
            }
        }
    ));
    assert!(error.source().is_some());
}

#[test]
fn sim_linkinds_reports_success_without_panic_wrapper() {
    let tt = TensorTrain::new(vec![site_tensor(2, vec![1.0, 2.0])]).unwrap();
    let simmed = tt.sim_linkinds().unwrap();

    assert_eq!(simmed.len(), 1);
    assert_eq!(
        simmed.tensor(0).unwrap().to_vec::<f64>().unwrap(),
        vec![1.0, 2.0]
    );
}
