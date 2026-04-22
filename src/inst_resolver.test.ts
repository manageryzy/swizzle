import { describe, expect, it } from 'vitest';
import {
  DEFAULT_CONFIG_PER_MODE,
  modeOfInst,
  resolveInst,
  shapesForMode,
  validShapesFor,
} from './inst_resolver';
import { INSTRUCTIONS } from './instructions';

describe('modeOfInst', () => {
  it('classifies every instruction', () => {
    for (const i of INSTRUCTIONS) {
      const m = modeOfInst(i);
      expect(['wmma', 'mma', 'wgmma', 'tcgen05']).toContain(m);
    }
  });
});

describe('resolveInst', () => {
  it('wgmma default resolves to m64n128k16 f32-acc variant', () => {
    const inst = resolveInst(DEFAULT_CONFIG_PER_MODE.wgmma);
    // The default config picks fp32 accumulator, which maps to the
    // `.f32.f16.f16` PTX mnemonic (cute atom SM90_64x128x16_F32F16F16_SS).
    expect(inst?.id).toBe('sm90.wgmma.m64n128k16.f32f16');
  });
  it('tcgen05 default resolves to an sm_100 m128n128k16 entry', () => {
    const inst = resolveInst(DEFAULT_CONFIG_PER_MODE.tcgen05);
    expect(inst).toBeDefined();
    expect(inst?.arch).toBe('sm100');
    expect(inst?.M).toBe(128);
    expect(inst?.N).toBe(128);
  });
  it('mma default resolves', () => {
    const inst = resolveInst(DEFAULT_CONFIG_PER_MODE.mma);
    expect(inst?.id).toContain('sm80.mma');
  });
  it('nonsense config returns undefined', () => {
    const inst = resolveInst({ ...DEFAULT_CONFIG_PER_MODE.wgmma, M: 999 });
    expect(inst).toBeUndefined();
  });
});

describe('shape enumerators', () => {
  it('wgmma PTX shapes cover every multiple of 8 from 8 to 256 at M=64, K=16', () => {
    // F7 expanded the wgmma catalog from the 8 CUTLASS atoms to the full
    // PTX-allowed set (N%8==0 in [8..256]). The atom-subset is still flagged
    // via `shapeSource: 'cutlass-atom'` on each InstSpec.
    const shapes = shapesForMode('wgmma').filter((s) => s.M === 64 && s.K === 16);
    expect(shapes.length).toBe(32);
  });
  it('tcgen05 shapes include M ∈ {32, 64, 128, 256}', () => {
    const shapes = shapesForMode('tcgen05');
    const Ms = new Set(shapes.map((s) => s.M));
    expect(Ms.has(32)).toBe(true);
    expect(Ms.has(256)).toBe(true);
  });
  it('validShapesFor(tcgen05 · fp16 · cg1 · dense) restricts K=16 only', () => {
    const shapes = validShapesFor('tcgen05', 'fp16', 'fp16', 'fp32', 1, false, false);
    expect(shapes.every((s) => s.K === 16)).toBe(true);
  });
});
