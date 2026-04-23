// Phase 6 tests — variant-specific WorldState: cluster (cg2), warps (.ws /
// non-.ws / coupled), auxiliary (sparse + block_scaled).
//
// Covers plan §B (WorldState.cluster, warps, auxiliary), §L1/§L2 (warp count
// and role transition), §L3 (warp-specialized concurrency matrix), §K6
// (cluster), §K7 (.ws), §K8 (sparse), §K9 (block_scaled).

import { describe, expect, it } from 'vitest';
import { INSTRUCTIONS, type InstSpec, type OperandSource } from './instructions';
import { simulate, warpsForFamily, type SimInput } from './simulation';

function findInst(id: string): InstSpec {
  const i = INSTRUCTIONS.find((x) => x.id === id);
  if (!i) throw new Error(`no InstSpec for ${id}`);
  return i;
}

function makeInput(patch: Partial<SimInput> & { inst: InstSpec }): SimInput {
  const base: SimInput = {
    inst: patch.inst,
    majorA: 'K',
    majorB: 'K',
    swizzle: '128B',
    dtypeA: patch.inst.aDtypes[0],
    dtypeB: patch.inst.bDtypes[0],
    accDtype: patch.inst.accDtypes[0],
    aSource: (patch.inst.aSource[0] ?? 'smem') as OperandSource,
    blkMMult: 1,
    blkNMult: 1,
    tileK: patch.inst.K * 4,
    kStages: 3,
    problemMMult: 4,
    problemNMult: 4,
    problemKMult: 4,
  };
  return { ...base, ...patch };
}

describe('Phase 6 — warpsForFamily (plan §L1)', () => {
  it('wgmma → 4 warps (cooperative warpgroup)', () => {
    expect(warpsForFamily('wgmma')).toBe(4);
  });
  it('tcgen05 → 4 warps', () => {
    expect(warpsForFamily('tcgen05')).toBe(4);
  });
  it('tcgen05.block_scaled → 4 warps', () => {
    expect(warpsForFamily('tcgen05.block_scaled')).toBe(4);
  });
  it('mma → 1 warp', () => {
    expect(warpsForFamily('mma')).toBe(1);
  });
  it('wmma → 1 warp', () => {
    expect(warpsForFamily('wmma')).toBe(1);
  });
});

describe('Phase 6 — cluster (cg2 only)', () => {
  it('cg1 inst: world.cluster === null', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(0);
    expect(w.cluster).toBeNull();
  });

  it('cg2 inst: world.cluster populated with thisCtaRole=leader', () => {
    const inst = findInst('sm100.tcgen05.cg2.m128n128k16.f16');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(0);
    expect(w.cluster).not.toBeNull();
    expect(w.cluster!.thisCtaRole).toBe('leader');
  });

  it('cg2: peerActive true during any active phase', () => {
    const inst = findInst('sm100.tcgen05.cg2.m128n128k16.f16');
    const r = simulate(makeInput({ inst }));
    // Sample a point early in the timeline when producer is firing.
    const firstProd = r.streams.producer[0];
    const midTick = (firstProd.startTick + firstProd.endTick) / 2;
    const w = r.worldAt(midTick);
    expect(w.cluster).not.toBeNull();
    expect(w.cluster!.peerActive).toBe(true);
  });

  it('cg2: sharedLoad true iff active producer is tma', () => {
    const inst = findInst('sm100.tcgen05.cg2.m128n128k16.f16');
    const r = simulate(makeInput({ inst }));
    const tmaProd = r.streams.producer.find((p) => p.producerSub === 'tma');
    expect(tmaProd).toBeTruthy();
    const mid = (tmaProd!.startTick + tmaProd!.endTick) / 2;
    const w = r.worldAt(mid);
    expect(w.cluster!.sharedLoad).toBe(true);
  });
});

describe('Phase 6 — warps (non-.ws warpgroup: all 4 share role)', () => {
  it('wgmma SS: all 4 warps share role during producer tma', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const tmaProd = r.streams.producer.find((p) => p.producerSub === 'tma');
    expect(tmaProd).toBeTruthy();
    const mid = (tmaProd!.startTick + tmaProd!.endTick) / 2;
    const w = r.worldAt(mid);
    expect(w.warps).toHaveLength(4);
    // During tma producer AND (possibly) consumer overlap, non-.ws
    // warpgroup collapses to consumer if any consumer is active, else
    // producer. Check that all 4 share the same role.
    const roles = new Set(w.warps.map((x) => x.role));
    expect(roles.size).toBe(1);
  });

  it('wgmma SS: during consumer phase, all 4 warps = consumer', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const cons = r.streams.consumer[0];
    const mid = (cons.startTick + cons.endTick) / 2;
    const w = r.worldAt(mid);
    for (const warp of w.warps) expect(warp.role).toBe('consumer');
  });
});

describe('Phase 6 — warps (.ws: warp 0 producer / warps 1-3 consumer)', () => {
  // .ws f16 inst ids: sm100.tcgen05.cg1.m{32,64,128}n{64,128,256}k16.f16.ws
  const wsId = 'sm100.tcgen05.cg1.m64n128k16.f16.ws';

  it('.ws inst exists in INSTRUCTIONS', () => {
    const inst = INSTRUCTIONS.find((x) => x.id === wsId);
    expect(inst).toBeTruthy();
    expect(inst!.warpSpecialized).toBe(true);
  });

  it('.ws: warp 0 = producer during tma producer phase', () => {
    const inst = findInst(wsId);
    const r = simulate(makeInput({ inst }));
    const tmaProd = r.streams.producer.find((p) => p.producerSub === 'tma');
    expect(tmaProd).toBeTruthy();
    const mid = (tmaProd!.startTick + tmaProd!.endTick) / 2;
    const w = r.worldAt(mid);
    expect(w.warps[0].role).toBe('producer');
  });

  it('.ws: warps 1..3 = consumer when consumer active (overlap w/ producer)', () => {
    const inst = findInst(wsId);
    const r = simulate(makeInput({ inst }));
    // Find the first tick where both producer tma and consumer are active.
    const cons = r.streams.consumer[0];
    // Consumer starts after slab 0 tma; find a producer phase that overlaps.
    let overlapTick = -1;
    for (const p of r.streams.producer) {
      if (p.producerSub !== 'tma') continue;
      const t0 = Math.max(p.startTick, cons.startTick);
      const t1 = Math.min(p.endTick, cons.endTick);
      if (t0 < t1) {
        overlapTick = (t0 + t1) / 2;
        break;
      }
    }
    if (overlapTick < 0) {
      // No overlap in this config — check just the consumer window.
      overlapTick = (cons.startTick + cons.endTick) / 2;
    }
    const w = r.worldAt(overlapTick);
    for (let wi = 1; wi < w.warps.length; wi++) {
      expect(w.warps[wi].role).toBe('consumer');
    }
  });
});

describe('Phase 6 — warps (single-warp families)', () => {
  it('sm80 mma: world.warps.length === 1', () => {
    const inst = findInst('sm80.mma.m16n8k16.f16');
    const r = simulate(makeInput({ inst, aSource: 'rmem' }));
    const w = r.worldAt(0);
    expect(w.warps).toHaveLength(1);
  });

  it('sm70 wmma: world.warps.length === 1', () => {
    const inst = findInst('sm70.wmma.m16n16k16.f16');
    const r = simulate(makeInput({ inst, aSource: 'gmem-wmma' }));
    const w = r.worldAt(0);
    expect(w.warps).toHaveLength(1);
  });

  it('sm80 mma: single warp role follows stream', () => {
    const inst = findInst('sm80.mma.m16n8k16.f16');
    const r = simulate(makeInput({ inst, aSource: 'rmem' }));
    const cons = r.streams.consumer[0];
    const mid = (cons.startTick + cons.endTick) / 2;
    const w = r.worldAt(mid);
    expect(w.warps[0].role).toBe('consumer');
  });
});

describe('Phase 6 — auxiliary (sparse + block_scaled)', () => {
  it('dense wgmma: all auxiliary flags false', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(0);
    expect(w.auxiliary.metadata).toBe(false);
    expect(w.auxiliary.scaleA).toBe(false);
    expect(w.auxiliary.scaleB).toBe(false);
  });

  it('sparse tcgen05: auxiliary.metadata === true', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16.sp');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(0);
    expect(w.auxiliary.metadata).toBe(true);
    expect(w.auxiliary.scaleA).toBe(false);
    expect(w.auxiliary.scaleB).toBe(false);
  });

  it('block_scaled tcgen05: auxiliary.scaleA + scaleB === true', () => {
    const inst = findInst('sm100.tcgen05.cg1.m128n128k32.mxf8f6f4');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(0);
    expect(w.auxiliary.scaleA).toBe(true);
    expect(w.auxiliary.scaleB).toBe(true);
    // Dense (not .sp): metadata stays false.
    expect(w.auxiliary.metadata).toBe(false);
  });
});

describe('Phase 6 — fragment stub', () => {
  it('all warps start with fragment.kind === "none"', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(0);
    for (const warp of w.warps) {
      expect(warp.fragment.kind).toBe('none');
    }
  });
});
