# tensor4all-aci

## src/batch.rs

### `pub fn new(values: & 'a [T], n_inputs: usize, n_points: usize) -> Result < Self >` (impl ElementwiseBatch < 'a , T >)

Creates a borrowed column-major batch view. `values` must contain exactly `n_inputs * n_points` entries in column-major order. Both `n_inputs` and `n_points` must be nonzero.

### `pub fn n_inputs(&self) -> usize` (impl ElementwiseBatch < 'a , T >)

Returns the number of operator inputs per interpolation point.

### `pub fn n_points(&self) -> usize` (impl ElementwiseBatch < 'a , T >)

Returns the number of interpolation points in the batch.

### `pub fn get(&self, input: usize, point: usize) -> Result < T >` (impl ElementwiseBatch < 'a , T >)

Returns one value using column-major indexing. The returned value is `values[input + n_inputs * point]`, so `input` varies fastest in the flat buffer.

### `pub fn as_col_major_slice(&self) -> & 'a [T]` (impl ElementwiseBatch < 'a , T >)

Returns the borrowed flat slice in column-major input/point layout.

## src/elementwise.rs

### `pub fn elementwise_batched(op: F, inputs: & [SimpleTensorTrain < T >], options: & AciOptions < T >) -> Result < AciResult < T > >`

Runs batched elementwise ACI over tensor-train inputs. This function approximates the pointwise application of `op` to `inputs`. The callback receives batches in column-major input/point layout through

### `pub fn elementwise(op: F, inputs: & [SimpleTensorTrain < T >], options: & AciOptions < T >) -> Result < AciResult < T > >`

Runs scalar elementwise ACI over tensor-train inputs. This convenience wrapper evaluates `op` once per interpolation point. The callback receives one value from each input tensor train in input order and

