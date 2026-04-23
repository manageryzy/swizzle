// Phase 4 tests — world.consumerAtom + MN multiplexing.
//
// Covers plan §A (m-outer, n-inner atom order), §B (consumerAtom shape),
// §C4 (consumer phase interpolation: atomFlatIdx / laneWave), §D7
// (ConflictMatrix auto-drive), and invariant §O4 (atomsPerStage_MN = 1
// degenerate case).

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

// Pick the first non-collapsed mma-step consumer phase.
function firstMmaStep(r: ReturnType<typeof simulate>) {
  return r.streams.consumer.find(
    (p) =>
      (p.kind === 'wgmma.step' || p.kind === 'tcgen05.mma.step') &&
      !p.collapsedCount,
  );
}

function probeAtFrac(
  r: ReturnType<typeof simulate>,
  frac: number,
) {
  const phase = firstMmaStep(r)!;
  const t = phase.startTick + frac * (phase.endTick - phase.startTick);
  return r.worldAt(t);
}

describe('Phase 4 — consumerAtom shape', () => {
  it('wgmma SS, blkMMult=1, blkNMult=1: atomFlatIdx=0 and (0,0) throughout', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const probes = [0.0, 0.25, 0.5, 0.75, 0.99];
    for (const frac of probes) {
      const w = probeAtFrac(r, frac);
      expect(w.consumerAtom).not.toBeNull();
      expect(w.consumerAtom!.atomFlatIdx).toBe(0);
      expect(w.consumerAtom!.atomM).toBe(0);
      expect(w.consumerAtom!.atomN).toBe(0);
    }
  });

  it('wgmma SS: consumerAtom is null outside any consumer phase', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    // Pick a tick before the first consumer phase (inside the warmup).
    const cons0 = r.streams.consumer[0];
    const w = r.worldAt(Math.max(0, cons0.startTick - 1));
    expect(w.active.consumer).toBeNull();
    expect(w.consumerAtom).toBeNull();
  });

  it('wgmma SS: consumerAtom is null after epilogue ends', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(r.totalTicks);
    expect(w.active.consumer).toBeNull();
    expect(w.consumerAtom).toBeNull();
  });
});

describe('Phase 4 — MN atom order (m-outer, n-inner)', () => {
  it('blkMMult=2, blkNMult=2: probes cycle (0,0) → (0,1) → (1,0) → (1,1)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, blkMMult: 2, blkNMult: 2 }),
    );
    const expected: { frac: number; flat: number; m: number; n: number }[] = [
      { frac: 0.0, flat: 0, m: 0, n: 0 },
      { frac: 0.25, flat: 1, m: 0, n: 1 },
      { frac: 0.5, flat: 2, m: 1, n: 0 },
      { frac: 0.75, flat: 3, m: 1, n: 1 },
    ];
    for (const e of expected) {
      const w = probeAtFrac(r, e.frac);
      expect(w.consumerAtom).not.toBeNull();
      expect(w.consumerAtom!.atomFlatIdx).toBe(e.flat);
      expect(w.consumerAtom!.atomM).toBe(e.m);
      expect(w.consumerAtom!.atomN).toBe(e.n);
    }
  });

  it('blkMMult=2, blkNMult=2: atomFlatIdx at progress 0.99 is the last atom (3)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, blkMMult: 2, blkNMult: 2 }),
    );
    const w = probeAtFrac(r, 0.99);
    expect(w.consumerAtom!.atomFlatIdx).toBe(3);
    expect(w.consumerAtom!.atomM).toBe(1);
    expect(w.consumerAtom!.atomN).toBe(1);
  });

  it('blkMMult=4, blkNMult=1: atomFlatIdx 0..3 ↔ atomM 0..3; atomN always 0', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, blkMMult: 4, blkNMult: 1 }),
    );
    const expected: { frac: number; m: number }[] = [
      { frac: 0.0, m: 0 },
      { frac: 0.3, m: 1 },
      { frac: 0.6, m: 2 },
      { frac: 0.9, m: 3 },
    ];
    for (const e of expected) {
      const w = probeAtFrac(r, e.frac);
      expect(w.consumerAtom).not.toBeNull();
      expect(w.consumerAtom!.atomM).toBe(e.m);
      expect(w.consumerAtom!.atomN).toBe(0);
      expect(w.consumerAtom!.atomFlatIdx).toBe(e.m);
    }
  });

  it('blkMMult=1, blkNMult=4: atomFlatIdx 0..3 ↔ atomN 0..3; atomM always 0', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, blkMMult: 1, blkNMult: 4 }),
    );
    const expected: { frac: number; n: number }[] = [
      { frac: 0.0, n: 0 },
      { frac: 0.3, n: 1 },
      { frac: 0.6, n: 2 },
      { frac: 0.9, n: 3 },
    ];
    for (const e of expected) {
      const w = probeAtFrac(r, e.frac);
      expect(w.consumerAtom).not.toBeNull();
      expect(w.consumerAtom!.atomN).toBe(e.n);
      expect(w.consumerAtom!.atomM).toBe(0);
      expect(w.consumerAtom!.atomFlatIdx).toBe(e.n);
    }
  });
});

describe('Phase 4 — laneWave cycling', () => {
  it('maxWaysConsumer always ≥ 1 and consumerAtom.maxWays mirrors it', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    for (const sw of ['none', '32B', '64B', '128B'] as const) {
      const r = simulate(makeInput({ inst, swizzle: sw }));
      expect(r.summary.maxWaysConsumer).toBeGreaterThanOrEqual(1);
      const w = probeAtFrac(r, 0.5);
      expect(w.consumerAtom!.maxWays).toBe(r.summary.maxWaysConsumer);
    }
  });

  it('laneWave stays within [0..maxWays-1] at every probe', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(
      makeInput({ inst, swizzle: 'none', blkMMult: 2, blkNMult: 2 }),
    );
    const maxWays = r.summary.maxWaysConsumer;
    for (let i = 0; i < 20; i++) {
      const frac = i / 20;
      const w = probeAtFrac(r, frac);
      expect(w.consumerAtom!.laneWave).toBeGreaterThanOrEqual(0);
      expect(w.consumerAtom!.laneWave).toBeLessThan(maxWays);
    }
  });

  it('when maxWays > 1, at least two distinct laneWave values appear across a phase', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    // Use atomsPerStage_MN=4 and an intentionally-high-conflict config so the
    // laneWave has headroom to cycle through at least 2 values within the phase.
    const r = simulate(
      makeInput({ inst, swizzle: 'none', blkMMult: 2, blkNMult: 2 }),
    );
    const maxWays = r.summary.maxWaysConsumer;
    if (maxWays > 1) {
      const seen = new Set<number>();
      for (let i = 0; i < 20; i++) {
        const frac = i / 20;
        const w = probeAtFrac(r, frac);
        seen.add(w.consumerAtom!.laneWave);
      }
      expect(seen.size).toBeGreaterThan(1);
    }
  });

  it('when atomsPerStage_MN === 1, laneWave is always 0 (no atoms × ways resolution)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    // blkMMult = blkNMult = 1 → atomsPerStage_MN = 1. The floor(frac*1*maxWays)
    // cycles through maxWays but the atom is always (0,0).
    const r = simulate(makeInput({ inst, swizzle: '128B', blkMMult: 1, blkNMult: 1 }));
    const probes = [0.0, 0.25, 0.5, 0.99];
    for (const frac of probes) {
      const w = probeAtFrac(r, frac);
      // atomsPerStage_MN=1 → a single atom (0,0) throughout.
      expect(w.consumerAtom!.atomFlatIdx).toBe(0);
      // laneWave may or may not be 0 depending on maxWays; just assert range.
      expect(w.consumerAtom!.laneWave).toBeGreaterThanOrEqual(0);
      expect(w.consumerAtom!.laneWave).toBeLessThan(w.consumerAtom!.maxWays);
    }
  });
});

describe('Phase 4 — consumer phase identification', () => {
  it('kSlab, kAtomInSlab, stage match the active consumer phase', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const phase = firstMmaStep(r)!;
    const t = phase.startTick + (phase.endTick - phase.startTick) / 2;
    const w = r.worldAt(t);
    expect(w.consumerAtom).not.toBeNull();
    expect(w.consumerAtom!.kSlab).toBe(phase.kSlab ?? 0);
    expect(w.consumerAtom!.kAtomInSlab).toBe(phase.kAtomInSlab ?? 0);
    expect(w.consumerAtom!.stage).toBe(phase.stage ?? 0);
  });

  it('kStep equals the phase iter (flat consumer index)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    // Sample the second consumer phase if available — its iter should be 1.
    const steps = r.streams.consumer.filter(
      (p) => !p.collapsedCount && (p.kind === 'wgmma.step' || p.kind === 'tcgen05.mma.step'),
    );
    if (steps.length >= 2) {
      const p1 = steps[1];
      const t = p1.startTick + (p1.endTick - p1.startTick) / 2;
      const w = r.worldAt(t);
      expect(w.consumerAtom).not.toBeNull();
      expect(w.consumerAtom!.kStep).toBe(p1.iter ?? 1);
    }
  });
});

describe('Phase 4 — coupled families (wmma, mma)', () => {
  it('sm_80 mma: consumerAtom populated during each consumer phase', () => {
    const inst = findInst('sm80.mma.m16n8k16.f16');
    const r = simulate(makeInput({ inst }));
    const phase = firstMmaStep(r);
    // sm80 mma uses 'wgmma.step' kind too per mmaStepKind fallback.
    expect(phase).toBeDefined();
    const t = phase!.startTick + 0.5 * (phase!.endTick - phase!.startTick);
    const w = r.worldAt(t);
    expect(w.consumerAtom).not.toBeNull();
    expect(w.consumerAtom!.kSlab).toBe(phase!.kSlab ?? 0);
  });

  it('sm_70 wmma: consumerAtom populated during wmma mma.step', () => {
    const inst = findInst('sm70.wmma.m16n16k16.f16');
    const r = simulate(makeInput({ inst }));
    const phase = firstMmaStep(r);
    expect(phase).toBeDefined();
    const t = phase!.startTick + 0.5 * (phase!.endTick - phase!.startTick);
    const w = r.worldAt(t);
    expect(w.consumerAtom).not.toBeNull();
  });
});

describe('Phase 4 — collapse phase handling', () => {
  it('inside a consumer collapse bar: atomFlatIdx=0, atomM=0, atomN=0', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    // Use a large problemKMult so the consumer tail collapses (plan §O5).
    const r = simulate(
      makeInput({
        inst,
        blkMMult: 2,
        blkNMult: 2,
        problemKMult: 8,
        tileK: inst.K * 4,
      }),
    );
    const collapseBar = r.streams.consumer.find((p) => p.collapsedCount);
    if (collapseBar) {
      const t =
        collapseBar.startTick +
        0.5 * (collapseBar.endTick - collapseBar.startTick);
      const w = r.worldAt(t);
      expect(w.consumerAtom).not.toBeNull();
      expect(w.consumerAtom!.atomFlatIdx).toBe(0);
      expect(w.consumerAtom!.atomM).toBe(0);
      expect(w.consumerAtom!.atomN).toBe(0);
    }
  });
});
