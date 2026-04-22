import { describe, expect, it } from 'vitest';
import { INSTRUCTIONS } from './instructions';
import { tileDimsFor } from './tile_dims';

const W = INSTRUCTIONS.find((i) => i.id === 'sm90.wgmma.m64n128k16.f32f16')!;
const T100 = INSTRUCTIONS.find((i) => i.id === 'sm100.tcgen05.cg1.m128n128k16.f16')!;

describe('tileDimsFor', () => {
  it('wgmma.m64n128k16 fp16 K-major: A = 64 × 32B = 2 KiB', () => {
    const a = tileDimsFor(W, 'A', 'K');
    expect(a.rows).toBe(64);
    expect(a.rowStrideBytes).toBe(32); // 16 fp16 = 32B
    expect(a.tileBytes).toBe(2048);
  });

  it('wgmma.m64n128k16 fp16 MN-major A: rows = K = 16, rowStride = M * 2B = 128B', () => {
    const a = tileDimsFor(W, 'A', 'MN');
    expect(a.rows).toBe(16);
    expect(a.rowStrideBytes).toBe(128);
    expect(a.tileBytes).toBe(2048);
  });

  it('wgmma.m64n128k16 fp16 K-major: B = 128 × 32B = 4 KiB', () => {
    const b = tileDimsFor(W, 'B', 'K');
    expect(b.rows).toBe(128);
    expect(b.rowStrideBytes).toBe(32);
    expect(b.tileBytes).toBe(4096);
  });

  it('tcgen05 m128n128k16 fp16 K-major: A = 128 × 32B', () => {
    const a = tileDimsFor(T100, 'A', 'K');
    expect(a.rows).toBe(128);
    expect(a.rowStrideBytes).toBe(32);
  });

  it('fp4 narrows cell count — row = K * 0.5 bytes', () => {
    // PTX Table 41: mxf4nvf4 dense K=64, cg2 sparse K=128. Pick the sparse
    // variant so the 0.5 B/elem row-stride comes out to 64 B.
    const mxf4 = INSTRUCTIONS.find(
      (i) => i.id.includes('mxf4nvf4') && i.sparse && i.M === 256 && i.K === 128,
    );
    if (!mxf4) return;
    const a = tileDimsFor(mxf4, 'A', 'K');
    expect(a.rowStrideBytes).toBe(64);
  });
});
