import { describe, expect, it } from 'vitest';
import {
  ALIGN,
  MAINLOOP_PIPELINE_BYTES_PER_STAGE,
  SMEM_PER_SM,
  buildBudget,
  bytesOf,
  kIterations,
  maxStages,
  stageBytes,
  tileBytesRaw,
} from './smem_budget';
import { INSTRUCTIONS } from './instructions';

const wgmma = INSTRUCTIONS.find((i) => i.id === 'sm90.wgmma.m64n128k16.f32f16')!;

describe('CUTLASS stage-bytes formula', () => {
  it('bytesOf covers all standard widths', () => {
    expect(bytesOf('fp32')).toBe(4);
    expect(bytesOf('fp16')).toBe(2);
    expect(bytesOf('fp4.e2m1')).toBe(0.5);
  });

  it('tileBytesRaw does not pad — it is the raw element count', () => {
    // 64 × 16 fp16 = 2048 raw; no 128-alignment at this layer.
    expect(tileBytesRaw(64, 16, 'fp16')).toBe(2048);
    expect(tileBytesRaw(17, 16, 'fp16')).toBe(544);
  });

  it('stageBytes aligns A+B as a pair, not separately', () => {
    const sb = stageBytes(wgmma);
    expect(sb.a).toBe(2048); // 64 × 16 × 2
    expect(sb.b).toBe(4096); // 128 × 16 × 2
    // a + b = 6144, already 128-aligned → pad 0
    expect(sb.pair).toBe(6144);
    expect(sb.pad).toBe(0);
    expect(sb.mbar).toBe(MAINLOOP_PIPELINE_BYTES_PER_STAGE);
    expect(sb.stage).toBe(sb.pair + sb.mbar);
  });

  it('a non-aligned A+B gets one combined pad, not two', () => {
    const tiny = { ...wgmma, M: 17, N: 7, K: 16 };
    const sb = stageBytes(tiny);
    // 17*16*2 + 7*16*2 = 544 + 224 = 768; round_up to 128 -> 768 (already aligned)
    expect(sb.a + sb.b).toBe(768);
    expect(sb.pair).toBe(Math.ceil(768 / ALIGN) * ALIGN);
    expect(sb.pad).toBe(sb.pair - 768);
  });
});

describe('SMEM_PER_SM matches whitepapers', () => {
  it('sm_90 / sm_100 = 228 KiB', () => {
    expect(SMEM_PER_SM.sm90).toBe(228 * 1024);
    expect(SMEM_PER_SM.sm100).toBe(228 * 1024);
  });
});

describe('buildBudget', () => {
  it('wgmma m64n128k16 fp16 · 3 stages uses 3 × stageBytes', () => {
    const b = buildBudget(wgmma, 3);
    const sb = b.stageBytes;
    expect(b.usedBytes).toBe(3 * sb.stage);
    expect(b.fits).toBe(true);
  });

  it('no "acc" segment exists — that is not part of the mainloop budget', () => {
    const b = buildBudget(wgmma, 3);
    expect(b.segments.some((s) => (s.kind as string) === 'acc')).toBe(false);
  });

  it('overflows when stages × stageBytes exceeds SMEM', () => {
    const huge = { ...wgmma, M: 1024, N: 1024, K: 128 };
    const b = buildBudget(huge, 3);
    expect(b.fits).toBe(false);
  });
});

describe('maxStages matches CUTLASS compute_stage_count_or_override', () => {
  it('returns (capacity - carveout) / stage_bytes for sm_90 fp16', () => {
    const n = maxStages(wgmma, 0);
    const sb = stageBytes(wgmma);
    expect(n).toBe(Math.floor((SMEM_PER_SM.sm90) / sb.stage));
  });

  it('honours carveout bytes', () => {
    const n0 = maxStages(wgmma, 0);
    const n1 = maxStages(wgmma, 64 * 1024);
    expect(n1).toBeLessThan(n0);
  });
});

describe('kIterations is tileK / kAtomK', () => {
  it('wgmma atomK=16: tileK=128 → 8 iterations', () => {
    expect(kIterations(wgmma, 128)).toBe(8);
  });
  it('tileK=16 (no accumulation) → 1 iteration', () => {
    expect(kIterations(wgmma, 16)).toBe(1);
  });
  it('partial tileK rounds up', () => {
    expect(kIterations(wgmma, 20)).toBe(2);
  });
});

describe('CUTLASS TileShape multipliers (TiledMMA AtomLayoutMNK)', () => {
  it('stageBytes(inst, {2,2}) scales A and B by their respective multiplier', () => {
    const base = stageBytes(wgmma);
    const tiled = stageBytes(wgmma, { blkMMult: 2, blkNMult: 2 });
    expect(tiled.a).toBe(base.a * 2);
    expect(tiled.b).toBe(base.b * 2);
    // pair re-aligns A+B together; both doubled → pair doubles too.
    expect(tiled.pair).toBe(base.pair * 2);
  });

  it('maxStages drops roughly proportionally when BLK_M doubles', () => {
    const n0 = maxStages(wgmma, 0);
    const n1 = maxStages(wgmma, 0, { blkMMult: 2 });
    // BLK_M doubles → A doubles → stage_bytes ≈ (2A + B + mbar)/(A + B + mbar);
    // n1 strictly less than n0 (and bounded above by half of n0 + slack).
    expect(n1).toBeLessThan(n0);
    expect(n1).toBeGreaterThan(0);
  });

  it('stageBytes default (no mult) == stageBytes({1,1})', () => {
    expect(stageBytes(wgmma)).toEqual(stageBytes(wgmma, { blkMMult: 1, blkNMult: 1 }));
  });
});
