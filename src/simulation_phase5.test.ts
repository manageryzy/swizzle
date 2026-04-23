// Phase 5 tests — world.cTile accumulation + epilogue sweeps.
//
// Covers plan §C4 (consumer phase .end effect increments accumulated[m][n]
// for every (m, n)), §N invariant 7 (monotone, caps at slabCount ×
// atomsPerStage_K), §O5 (collapse-phase handling: bump by collapsedCount),
// and §D9/§D2/§D4 rendering expectations (mid-phase interpolation gives
// visible ramp inside the collapse bar and regular phases).

import { describe, expect, it } from 'vitest';
import { INSTRUCTIONS, type InstSpec, type OperandSource } from './instructions';
import { simulate, type SimInput } from './simulation';

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

describe('Phase 5 — summary.maxAccumulatedPerAtom', () => {
  it('default wgmma SS config: slabCount × atomsPerStage_K = 16', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    expect(r.summary.slabCount).toBe(4);
    expect(r.summary.atomsPerStage_K).toBe(4);
    expect(r.summary.maxAccumulatedPerAtom).toBe(16);
    expect(r.summary.maxAccumulatedPerAtom).toBe(
      r.summary.slabCount * r.summary.atomsPerStage_K,
    );
  });

  it('matches consumerItersTotal (= slabCount × atomsPerStage_K)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, problemKMult: 2 }));
    expect(r.summary.maxAccumulatedPerAtom).toBe(r.summary.consumerItersTotal);
  });
});

describe('Phase 5 — cTile shape + initial state', () => {
  it('cTile is a [blkMMult][blkNMult] 2-D grid of zeros at tick 0', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 3 }));
    const w = r.worldAt(0);
    expect(w.cTile).not.toBeNull();
    expect(w.cTile!.accumulated.length).toBe(2);
    expect(w.cTile!.accumulated[0].length).toBe(3);
    for (let m = 0; m < 2; m++) {
      for (let n = 0; n < 3; n++) {
        expect(w.cTile!.accumulated[m][n]).toBe(0);
        expect(w.cTile!.epilogueStaged[m][n]).toBe(0);
        expect(w.cTile!.epilogueDrained[m][n]).toBe(0);
      }
    }
  });

  it('default blkMMult=blkNMult=1: cTile is 1×1', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(0);
    expect(w.cTile!.accumulated.length).toBe(1);
    expect(w.cTile!.accumulated[0].length).toBe(1);
    expect(w.cTile!.accumulated[0][0]).toBe(0);
  });
});

describe('Phase 5 — accumulation at phase ends (regular phases)', () => {
  it('after consumer phase 0 ends: accumulated[0][0] === 1', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const cons0 = r.streams.consumer[0];
    const w = r.worldAt(cons0.endTick);
    expect(w.cTile!.accumulated[0][0]).toBe(1);
  });

  it('after consumer phase 0 ends with blkMMult=2, blkNMult=2: all (m,n) === 1', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 2 }));
    const cons0 = r.streams.consumer[0];
    const w = r.worldAt(cons0.endTick);
    for (let m = 0; m < 2; m++) {
      for (let n = 0; n < 2; n++) {
        expect(w.cTile!.accumulated[m][n]).toBe(1);
      }
    }
  });

  it('after all slab-0 consumer phases: accumulated[0][0] === atomsPerStage_K', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const aK = r.summary.atomsPerStage_K;
    // The first aK consumer phases are slab 0's (flat order in the stream).
    const phase = r.streams.consumer[aK - 1];
    const w = r.worldAt(phase.endTick);
    expect(w.cTile!.accumulated[0][0]).toBe(aK);
  });

  it('at totalTicks: accumulated[m][n] === slabCount × atomsPerStage_K for every (m,n)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, blkMMult: 2, blkNMult: 2 }),
    );
    const w = r.worldAt(r.totalTicks);
    const expected = r.summary.slabCount * r.summary.atomsPerStage_K;
    for (let m = 0; m < 2; m++) {
      for (let n = 0; n < 2; n++) {
        expect(w.cTile!.accumulated[m][n]).toBe(expected);
      }
    }
  });
});

describe('Phase 5 — invariant 7: monotone non-decreasing + cap', () => {
  it('accumulated never decreases across any pair of ticks', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, blkMMult: 2, blkNMult: 2, problemKMult: 4 }),
    );
    let prev = r.worldAt(0).cTile!.accumulated.map((r) => r.slice());
    for (let t = 1; t <= r.totalTicks; t++) {
      const cur = r.worldAt(t).cTile!.accumulated;
      for (let m = 0; m < 2; m++) {
        for (let n = 0; n < 2; n++) {
          expect(cur[m][n]).toBeGreaterThanOrEqual(prev[m][n]);
        }
      }
      prev = cur.map((row) => row.slice());
    }
  });

  it('accumulated never exceeds maxAccumulatedPerAtom at any tick', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, blkMMult: 2, blkNMult: 2, problemKMult: 4 }),
    );
    const cap = r.summary.maxAccumulatedPerAtom;
    for (let t = 0; t <= r.totalTicks; t++) {
      const c = r.worldAt(t).cTile!.accumulated;
      for (let m = 0; m < 2; m++) {
        for (let n = 0; n < 2; n++) {
          expect(c[m][n]).toBeLessThanOrEqual(cap);
        }
      }
    }
  });
});

describe('Phase 5 — collapse-phase handling (§O5)', () => {
  it('problemKMult=8 (consumerItersTotal=32, > MAX=8): collapse bar bumps by collapsedCount', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({
        inst,
        blkMMult: 2,
        blkNMult: 2,
        problemKMult: 8,
        tileK: inst.K * 4,
      }),
    );
    // There should be a collapse bar in the consumer stream.
    const collapseBar = r.streams.consumer.find((p) => p.collapsedCount);
    expect(collapseBar).toBeDefined();
    const collapsedCount = collapseBar!.collapsedCount!;
    // Sample before the collapse bar and right after — the accumulated count
    // should jump by `collapsedCount` (since consumer phases before the
    // collapse each bumped by 1, and the collapse bumps by collapsedCount).
    const beforeTick = Math.max(0, collapseBar!.startTick - 1);
    const before = r.worldAt(beforeTick).cTile!.accumulated[0][0];
    const after = r.worldAt(collapseBar!.endTick).cTile!.accumulated[0][0];
    expect(after - before).toBe(collapsedCount);
  });

  it('at mainloop end (with collapse bar), total = slabCount × atomsPerStage_K', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({
        inst,
        blkMMult: 2,
        blkNMult: 2,
        problemKMult: 8,
        tileK: inst.K * 4,
      }),
    );
    const w = r.worldAt(r.totalTicks);
    const expected = r.summary.slabCount * r.summary.atomsPerStage_K;
    for (let m = 0; m < 2; m++) {
      for (let n = 0; n < 2; n++) {
        expect(w.cTile!.accumulated[m][n]).toBe(expected);
      }
    }
  });

  it('mid-collapse bar: accumulated advances (no flatline) via interpolation', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({
        inst,
        problemKMult: 8,
        tileK: inst.K * 4,
      }),
    );
    const collapseBar = r.streams.consumer.find((p) => p.collapsedCount);
    if (!collapseBar) return;
    const atStart = r.worldAt(collapseBar.startTick).cTile!.accumulated[0][0];
    const atMid = r.worldAt(
      Math.floor((collapseBar.startTick + collapseBar.endTick) / 2),
    ).cTile!.accumulated[0][0];
    const atEnd = r.worldAt(collapseBar.endTick).cTile!.accumulated[0][0];
    expect(atMid).toBeGreaterThanOrEqual(atStart);
    expect(atEnd).toBeGreaterThanOrEqual(atMid);
    // If collapsedCount >= 2, we should see at least some intra-bar progress.
    if (collapseBar.collapsedCount! >= 2) {
      expect(atEnd).toBeGreaterThan(atStart);
    }
  });
});

describe('Phase 5 — epilogue sweeps', () => {
  it('at start of epilogue.stg_smem: (0,0) > 0 while last cell is 0 (row-major sweep)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 2 }));
    const stg = r.streams.epilogue.find((p) => p.kind === 'epilogue.stg_smem');
    expect(stg).toBeDefined();
    // Sample very early into the phase (frac is just above 0).
    const probeTick = stg!.startTick + 1;
    const w = r.worldAt(probeTick);
    expect(w.cTile!.epilogueStaged[0][0]).toBeGreaterThan(0);
    // Last cell shouldn't be fully staged yet at the very start.
    expect(w.cTile!.epilogueStaged[1][1]).toBeLessThan(1);
  });

  it('at end of epilogue.stg_smem: all cells === 1', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 2 }));
    const stg = r.streams.epilogue.find((p) => p.kind === 'epilogue.stg_smem');
    expect(stg).toBeDefined();
    const w = r.worldAt(stg!.endTick);
    for (let m = 0; m < 2; m++) {
      for (let n = 0; n < 2; n++) {
        expect(w.cTile!.epilogueStaged[m][n]).toBe(1);
      }
    }
  });

  it('during epilogue.tma.store mid-phase: drain sweeps row-major', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 2 }));
    const store = r.streams.epilogue.find(
      (p) => p.kind === 'epilogue.tma.store',
    );
    expect(store).toBeDefined();
    const probeTick = store!.startTick + 1;
    const w = r.worldAt(probeTick);
    // (0, 0) drained > 0 at the very start.
    expect(w.cTile!.epilogueDrained[0][0]).toBeGreaterThan(0);
    // Last cell still 0 at the very start.
    expect(w.cTile!.epilogueDrained[1][1]).toBeLessThan(1);
  });

  it('at totalTicks: drain is fully 1 everywhere', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 2 }));
    const w = r.worldAt(r.totalTicks);
    for (let m = 0; m < 2; m++) {
      for (let n = 0; n < 2; n++) {
        expect(w.cTile!.epilogueDrained[m][n]).toBe(1);
      }
    }
  });

  it('before any epilogue: staged and drained are both 0', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 2 }));
    const stg = r.streams.epilogue.find((p) => p.kind === 'epilogue.stg_smem');
    expect(stg).toBeDefined();
    const w = r.worldAt(Math.max(0, stg!.startTick - 1));
    for (let m = 0; m < 2; m++) {
      for (let n = 0; n < 2; n++) {
        expect(w.cTile!.epilogueStaged[m][n]).toBe(0);
        expect(w.cTile!.epilogueDrained[m][n]).toBe(0);
      }
    }
  });
});

describe('Phase 5 — mid-phase interpolation inside a regular consumer phase', () => {
  it('blkMMult=2, blkNMult=2: atom (0,0) bumps first, (1,1) last', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, blkMMult: 2, blkNMult: 2 }));
    const phase = r.streams.consumer[0];
    // Right after phase 0 starts — only atom (0, 0) has been "hit" (atomFlatIdx=0).
    // Then (0,1), then (1,0), then (1,1).
    const probe = (frac: number) =>
      r.worldAt(
        Math.floor(phase.startTick + frac * (phase.endTick - phase.startTick)),
      );
    // At frac=0.5 with atomsMN=4: running = floor(0.5 * 4) = 2, so (0,0) and
    // (0,1) should be bumped, but not (1,0) or (1,1).
    const w = probe(0.5);
    expect(w.cTile!.accumulated[0][0]).toBe(1);
    expect(w.cTile!.accumulated[0][1]).toBe(1);
    expect(w.cTile!.accumulated[1][0]).toBe(0);
    expect(w.cTile!.accumulated[1][1]).toBe(0);
  });
});
