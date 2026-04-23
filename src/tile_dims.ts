// Tile dimensions derived from (instruction, operand, major-ness).
// Used by the SMEM panel to draw A and B at their actual geometry rather
// than a fixed 32×128B strip.

import type { InstSpec, Major } from './instructions';
import { bytesOf } from './smem_budget';

export interface TileDims {
  rows: number;
  cols: number; // cells (4B each)
  rowStrideBytes: number; // bytes per row
  tileBytes: number;
  elemBytes: number;
  dtypeLabel: string;
  outerLabel: string; // 'M' for A, 'N' for B
  innerLabel: string; // always 'K'
}

export function tileDimsFor(
  i: InstSpec,
  operand: 'A' | 'B',
  major: Major,
  outerMult: number = 1,
): TileDims {
  const dtype = operand === 'A' ? i.aDtypes[0] : i.bDtypes[0];
  const elemBytes = bytesOf(dtype);
  // `outerMult` is the BLK_M / BLK_N multiplier from the CUTLASS TiledMMA;
  // by default 1 (one atom per CTA, original behaviour).
  const outer = (operand === 'A' ? i.M : i.N) * Math.max(1, outerMult);
  const inner = i.K;
  const totalBytes = Math.ceil(outer * inner * elemBytes);

  if (major === 'K') {
    // K is the fast axis → rows span outer, K-elements per row.
    const rowStrideBytes = Math.max(4, Math.ceil(inner * elemBytes));
    return {
      rows: outer,
      cols: Math.max(1, Math.ceil(rowStrideBytes / 4)),
      rowStrideBytes,
      tileBytes: totalBytes,
      elemBytes,
      dtypeLabel: dtype,
      outerLabel: operand === 'A' ? 'M' : 'N',
      innerLabel: 'K',
    };
  }
  // MN-major: outer is the fast axis → rows span K.
  const rowStrideBytes = Math.max(4, Math.ceil(outer * elemBytes));
  return {
    rows: inner,
    cols: Math.max(1, Math.ceil(rowStrideBytes / 4)),
    rowStrideBytes,
    tileBytes: totalBytes,
    elemBytes,
    dtypeLabel: dtype,
    outerLabel: operand === 'A' ? 'M' : 'N',
    innerLabel: 'K',
  };
}
