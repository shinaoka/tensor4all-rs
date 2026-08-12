use tensor4all_core::IdxTensor;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn eager_tensor_and_idx_tensor_are_send_sync() {
    assert_send_sync::<tenferro_ad::EagerTensor>();
    assert_send_sync::<IdxTensor>();
}
