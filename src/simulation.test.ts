// Phase 1 smoke tests for `simulate()`. Covers plan §F1 items that do NOT
// require populated producerTransfer / consumerAtom / cTile state (those
// fields arrive in later phases). See plan §N for invariants.

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

describe('simulate — Phase 1 skeleton', () => {
  const wgmma = findInst('sm90.wgmma.m64n128k16.f32f16');
  const mma80 = findInst('sm80.mma.m16n8k16.f16');
  const wmma = findInst('sm70.wmma.m16n16k16.f16');

  it('wgmma_SS emits non-empty producer, consumer, epilogue streams', () => {
    const r = simulate(makeInput({ inst: wgmma }));
    expect(r.streams.producer.length).toBeGreaterThanOrEqual(1);
    expect(r.streams.consumer.length).toBeGreaterThanOrEqual(1);
    // wgmma epilogue has 2 phases (stg_smem → tma.store). tcgen05 has 3
    // (tcgen05.ld → stg_smem → tma.store). See pipeline_state.ts · emitTimeline.
    expect(r.streams.epilogue.length).toBeGreaterThanOrEqual(2);
  });

  it('tcgen05 SS epilogue stream has 3 phases (tcgen05.ld → stg_smem → tma.store)', () => {
    const tcg = findInst('sm100.tcgen05.cg1.m64n128k16.f16');
    const r = simulate(makeInput({ inst: tcg }));
    expect(r.streams.epilogue.length).toBeGreaterThanOrEqual(3);
    expect(r.streams.epilogue[0].kind).toBe('tcgen05.ld');
  });

  it('wgmma_SS worldAt(0) has producer active but not consumer (warmup)', () => {
    const r = simulate(makeInput({ inst: wgmma }));
    const w = r.worldAt(0);
    expect(w.active.producer).not.toBeNull();
    expect(w.active.consumer).toBeNull();
  });

  it('wgmma_SS steady-state tick shows overlapping producer and consumer', () => {
    // Warpspec's raison d'être: mid-mainloop, some tick has both streams live.
    const r = simulate(makeInput({ inst: wgmma }));
    let foundOverlap = false;
    // Sample ticks near the first consumer start to find the steady state.
    const firstConsumer = r.streams.consumer[0]?.startTick ?? 0;
    const lastConsumer = r.streams.consumer.at(-1)?.endTick ?? r.totalTicks;
    const step = Math.max(1, Math.floor((lastConsumer - firstConsumer) / 12));
    for (let t = firstConsumer; t < lastConsumer; t += step) {
      const w = r.worldAt(t);
      if (w.active.producer && w.active.consumer) {
        foundOverlap = true;
        break;
      }
    }
    expect(foundOverlap).toBe(true);
  });

  it('sm80.mma has familyShape "cpasync-mma" and no ring', () => {
    const r = simulate(makeInput({ inst: mma80 }));
    expect(r.summary.familyShape).toBe('cpasync-mma');
    expect(r.summary.hasRing).toBe(false);
    expect(r.summary.hasMbar).toBe(false);
  });

  it('sm70.wmma has familyShape "wmma-direct" and empty ring', () => {
    const r = simulate(makeInput({ inst: wmma }));
    expect(r.summary.familyShape).toBe('wmma-direct');
    const w = r.worldAt(0);
    expect(w.ring.length).toBe(0);
    expect(w.mbar.length).toBe(0);
  });

  it('wgmma_SS summary fields match BLK_K / atomK arithmetic', () => {
    // tileK = K*4, problemKMult = 4, blkMMult=blkNMult=1.
    const r = simulate(makeInput({ inst: wgmma, problemKMult: 4 }));
    expect(r.summary.slabCount).toBe(4);
    expect(r.summary.atomsPerStage_K).toBe(4); // tileK / K = (K*4)/K = 4
    expect(r.summary.atomsPerStage_MN).toBe(1);
    expect(r.summary.consumerItersTotal).toBe(16);
    expect(r.summary.cCells).toEqual({ m: 1, n: 1 });
  });

  it('wgmma_SS ring initial state: every stage starts empty', () => {
    const r = simulate(makeInput({ inst: wgmma, kStages: 3 }));
    const w = r.worldAt(-1); // before any effect
    expect(w.ring.length).toBe(3);
    for (const slot of w.ring) {
      expect(slot.role).toBe('empty');
      expect(slot.fillFrac).toBe(0);
    }
  });

  it('wgmma_SS totalTicks agrees between SimResult and epilogue endTick', () => {
    const r = simulate(makeInput({ inst: wgmma }));
    const lastEnd = r.streams.epilogue.at(-1)!.endTick;
    expect(r.totalTicks).toBe(lastEnd);
  });

  it('wgmma_SS progress values are always in [0, 1]', () => {
    const r = simulate(makeInput({ inst: wgmma }));
    const samples = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
    for (const f of samples) {
      const t = Math.floor(r.totalTicks * f);
      const w = r.worldAt(t);
      expect(w.progress.producer).toBeGreaterThanOrEqual(0);
      expect(w.progress.producer).toBeLessThanOrEqual(1);
      expect(w.progress.consumer).toBeGreaterThanOrEqual(0);
      expect(w.progress.consumer).toBeLessThanOrEqual(1);
      expect(w.progress.epilogue).toBeGreaterThanOrEqual(0);
      expect(w.progress.epilogue).toBeLessThanOrEqual(1);
    }
  });

  it('wgmma_SS ring fillFrac is monotone within a producer phase', () => {
    const r = simulate(makeInput({ inst: wgmma }));
    const p0 = r.streams.producer[0];
    const ticks = [p0.startTick, p0.startTick + 1, p0.startTick + 2, p0.endTick - 1];
    let prev = -1;
    for (const t of ticks) {
      const w = r.worldAt(t);
      const slot = w.ring[p0.stage!];
      expect(slot.role).toBe('fill');
      expect(slot.fillFrac).toBeGreaterThanOrEqual(prev);
      prev = slot.fillFrac;
    }
  });

  it('invariant 1 — ring occupancy is always ≤ kStages', () => {
    const r = simulate(makeInput({ inst: wgmma, kStages: 3 }));
    for (let t = 0; t <= r.totalTicks; t += 1) {
      const w = r.worldAt(t);
      const occupied = w.ring.filter(
        (r) => r.role === 'fill' || r.role === 'hold' || r.role === 'consume',
      ).length;
      expect(occupied).toBeLessThanOrEqual(3);
    }
  });

  it('invariant 4 — producer phase iters strictly ascend', () => {
    const r = simulate(makeInput({ inst: wgmma }));
    let prev = -1;
    for (const p of r.streams.producer) {
      if (p.iter === undefined) continue;
      expect(p.iter).toBeGreaterThanOrEqual(prev);
      prev = p.iter;
    }
  });

  it('invariant 6 — first consumer phase starts at/after first producer end', () => {
    const r = simulate(makeInput({ inst: wgmma }));
    expect(r.streams.consumer[0].startTick).toBeGreaterThanOrEqual(
      r.streams.producer[0].endTick,
    );
  });
});
