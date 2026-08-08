use super::*;
use crate::backend::File;

#[test]
fn test_tagset_to_string() {
    let tags = TagSet::from_str("Site,n=1").unwrap();
    let s = tagset_to_string(&tags);
    // Tags are sorted, so order may differ
    assert!(s.contains("Site"));
    assert!(s.contains("n=1"));
}

#[test]
fn negative_index_set_length_is_rejected() -> Result<()> {
    let path = std::env::temp_dir().join("tensor4all_hdf5_index_negative_length.h5");
    let path = path.to_string_lossy().to_string();

    let file = File::create(&path)?;
    let group = file.create_group("indices")?;
    write_index_set(&group, &[DynIndex::new_dyn(2)])?;
    group.dataset("length")?.as_writer().write_scalar(&-1_i64)?;
    drop(file);

    let file = File::open(&path)?;
    let error = read_index_set(&file.group("indices")?)
        .unwrap_err()
        .to_string();
    assert!(error.contains("length"));
    assert!(error.contains("-1"));

    std::fs::remove_file(path).ok();
    Ok(())
}
