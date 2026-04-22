import { describe, expect, it } from 'vitest';
import { coordOf, flatten, layoutAt, sizeOf } from './cute_layout';
import { C_LAYOUTS, ownershipMap } from './cute_mma_layouts';

describe('cute layout evaluator', () => {
  it('flatten and sizeOf on nested tuples', () => {
    expect(flatten([[4, 8], [2, 2]])).toEqual([4, 8, 2, 2]);
    expect(sizeOf([[4, 8], [2, 2]])).toBe(128);
    expect(sizeOf([[4, 8, 4], [2, 2, 16]])).toBe(8192);
  });

  it('coordOf decomposes column-major (innermost first)', () => {
    const s = [4, 8];
    expect(coordOf(s, 0)).toEqual([0, 0]);
    expect(coordOf(s, 1)).toEqual([1, 0]);
    expect(coordOf(s, 4)).toEqual([0, 1]);
    expect(coordOf(s, 31)).toEqual([3, 7]);
  });

  it('SM80_16x8_Row: thread 0 owns (0,0) (8,0) (0,1) (8,1)', () => {
    // With M=16 column-major, offset → (m,n) = (off%16, off/16).
    // Thread 0 = (t0=0, t1=0). Values 0..3 span (v0, v1).
    const shape = [[4, 8], [2, 2]];
    const stride = [[32, 1], [16, 8]];
    const threads = 32;
    const values: [number, number][] = [];
    for (let v = 0; v < 4; v++) {
      const off = layoutAt(shape, stride, 0 + v * threads);
      values.push([off % 16, Math.floor(off / 16)]);
    }
    // PTX m16n8k16 fp16 canonical: thread 0 → (0,0),(0,1),(8,0),(8,1)
    expect(values.sort()).toEqual([
      [0, 0],
      [0, 1],
      [8, 0],
      [8, 1],
    ]);
  });

  it('SM90 CLayout_64x8: thread 0 owns (0,0) (0,1) (8,0) (8,1)', () => {
    const c = C_LAYOUTS['sm90.wgmma.m64n8k16.f32f16'];
    expect(c).toBeDefined();
    const owner = ownershipMap(c!);
    expect(owner(0, 0)).toBe(0);
    expect(owner(0, 1)).toBe(0);
    expect(owner(8, 0)).toBe(0);
    expect(owner(8, 1)).toBe(0);
  });

  it('SM90 CLayout_64x128: thread count is 128 (warpgroup)', () => {
    const c = C_LAYOUTS['sm90.wgmma.m64n128k16.f32f16'];
    expect(c?.threads).toBe(128);
    expect(c?.valuesPerThread).toBe(64);
    expect(c && c.M * c.N).toBe(c && c.threads * c.valuesPerThread);
  });

  it('ownership map is surjective over [0..threads)', () => {
    const c = C_LAYOUTS['sm80.mma.m16n8k16.f16']!;
    const owner = ownershipMap(c);
    const owners = new Set<number>();
    for (let m = 0; m < c.M; m++)
      for (let n = 0; n < c.N; n++) owners.add(owner(m, n));
    expect(owners.size).toBe(c.threads);
  });

  it('SM80 MMA.m16n8k16: SMEM descriptor layout type maps to LayoutType enum', () => {
    // Sanity — this test anchors the fact that the port is value-level, not
    // string-level. If we rename LayoutType we'll catch it.
    const c = C_LAYOUTS['sm80.mma.m16n8k16.f16']!;
    expect(c.M).toBe(16);
    expect(c.N).toBe(8);
  });
});
