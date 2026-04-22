// Minimal cute::Layout evaluator.
//
// A cute Layout is (Shape, Stride) where both are hierarchical IntTuples.
// Evaluation rule (from cute docs and /tmp/cutlass/include/cute/layout.hpp):
//   layout(idx) = sum_i ( coord_i(idx) * stride_i )
// where coord_i decomposes idx into the flat shape COLUMN-MAJOR: the leftmost
// (innermost) dim cycles fastest.
//
// We represent Shape/Stride as arbitrarily nested number arrays. The exact
// nesting doesn't change evaluation because we flatten before computing —
// nesting is preserved only for readability and when we need to decompose
// "thread dim" vs "value dim" separately.

export type IntTuple = number | readonly IntTuple[];

export function isInt(t: IntTuple): t is number {
  return typeof t === 'number';
}

export function flatten(t: IntTuple): number[] {
  if (isInt(t)) return [t];
  return t.flatMap(flatten);
}

export function sizeOf(t: IntTuple): number {
  return flatten(t).reduce((a, b) => a * b, 1);
}

// idx ∈ [0, sizeOf(shape)) → flat coord list (column-major decomposition).
export function coordOf(shape: IntTuple, idx: number): number[] {
  const flat = flatten(shape);
  const coords: number[] = [];
  let rem = idx;
  for (const d of flat) {
    coords.push(rem % d);
    rem = Math.floor(rem / d);
  }
  return coords;
}

// Layout evaluation.
export function layoutAt(shape: IntTuple, stride: IntTuple, idx: number): number {
  const coord = coordOf(shape, idx);
  const strideFlat = flatten(stride);
  if (coord.length !== strideFlat.length) {
    throw new Error(
      `shape and stride have different flat arity: ${coord.length} vs ${strideFlat.length}`,
    );
  }
  let off = 0;
  for (let i = 0; i < coord.length; i++) off += coord[i] * strideFlat[i];
  return off;
}

// Convenience: enumerate every (idx → offset) pair.
export function* layoutPairs(shape: IntTuple, stride: IntTuple): Generator<[number, number]> {
  const n = sizeOf(shape);
  for (let i = 0; i < n; i++) yield [i, layoutAt(shape, stride, i)];
}
