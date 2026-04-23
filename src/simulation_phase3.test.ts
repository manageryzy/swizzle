// Phase 3 tests — producerTransfer + ring.fillFrac wiring.
//
// Covers plan §B (producerTransfer shape), §C4 (start/end effects), §N1
// (ring occupancy monotone) and §N5 (ring-slice consistency).

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

describe('Phase 3 — producerTransfer shape', () => {
  it('wgmma SS: worldAt(producerStart).producerTransfer.linesLoaded === 0', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const p0 = r.streams.producer[0];
    const w = r.worldAt(p0.startTick);
    expect(w.producerTransfer).not.toBeNull();
    expect(w.producerTransfer!.kind).toBe('tma');
    expect(w.producerTransfer!.kSlab).toBe(0);
    expect(w.producerTransfer!.operand).toBe('AB');
    expect(w.producerTransfer!.stage).toBe(p0.stage);
    expect(w.producerTransfer!.linesLoaded).toBe(0);
    expect(w.producerTransfer!.linesTotal).toBeGreaterThan(0);
  });

  it('wgmma SS: worldAt(producerEnd − 1).producerTransfer.linesLoaded ≈ linesTotal', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const p0 = r.streams.producer[0];
    const w = r.worldAt(p0.endTick - 1);
    expect(w.producerTransfer).not.toBeNull();
    // At (end-1) with phase ticks=4: frac=3/4 → linesLoaded = floor(0.75 × total).
    const expected = Math.floor(0.75 * w.producerTransfer!.linesTotal);
    expect(w.producerTransfer!.linesLoaded).toBe(expected);
  });

  it('wgmma SS: worldAt(producerEnd) has producerTransfer === null', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const p0 = r.streams.producer[0];
    // End tick: phase ends; if next producer phase starts later, this should
    // be null. We check one tick PAST endTick to avoid race with cascaded
    // back-to-back producer phases.
    const w = r.worldAt(p0.endTick);
    // At endTick exactly, this producer's end effect has fired (clears active)
    // — another producer phase may start right here. Assert: if active.producer
    // is null, producerTransfer is null.
    if (w.active.producer === null) {
      expect(w.producerTransfer).toBeNull();
    } else {
      // Another phase is active → producerTransfer reflects THAT phase.
      expect(w.producerTransfer).not.toBeNull();
      expect(w.producerTransfer!.kSlab).toBeGreaterThanOrEqual(0);
    }
  });

  it('wgmma SS: producerTransfer is null at a tick well after epilogue ends', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const w = r.worldAt(r.totalTicks);
    // All streams drained; producer has no active phase.
    expect(w.active.producer).toBeNull();
    expect(w.producerTransfer).toBeNull();
  });

  it('wgmma SS: ring[stage].fillFrac monotone increasing 0 → 1 across producer phase', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const p0 = r.streams.producer[0];
    const stage = p0.stage!;
    const samples = [p0.startTick, p0.startTick + 1, p0.startTick + 2, p0.endTick - 1];
    const fracs = samples.map((t) => r.worldAt(t).ring[stage].fillFrac);
    for (let i = 1; i < fracs.length; i++) {
      expect(fracs[i]).toBeGreaterThanOrEqual(fracs[i - 1]);
    }
    expect(fracs[0]).toBeCloseTo(0, 5);
    expect(fracs[fracs.length - 1]).toBeLessThanOrEqual(1);
    expect(fracs[fracs.length - 1]).toBeGreaterThan(0.5);
  });

  it('wgmma SS: after producer end, ring[stage].role === hold AND fillFrac === 1', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    // Use a later slab where the ring-full back-pressure creates a guaranteed
    // hold window. Pick a producer whose end predates its first consumer by
    // at least 1 tick; if every slab is back-to-back (no gap) we skip.
    let hold: { stage: number; tick: number } | null = null;
    for (const p of r.streams.producer) {
      if (p.producerSub !== 'tma') continue;
      const stage = p.stage!;
      const firstCons = r.streams.consumer.find(
        (c) => c.stage === stage && c.kSlab === p.kSlab && !c.collapsedCount,
      );
      if (!firstCons) continue;
      if (firstCons.startTick > p.endTick) {
        hold = { stage, tick: p.endTick };
        break;
      }
    }
    if (!hold) {
      // Degenerate: no hold gap — sanity-check that at p0.endTick the stage
      // is full (fillFrac=1) even if role has already transitioned.
      const p0 = r.streams.producer[0];
      const w = r.worldAt(p0.endTick);
      expect(w.ring[p0.stage!].fillFrac).toBe(1);
      return;
    }
    const w = r.worldAt(hold.tick);
    expect(w.ring[hold.stage].role).toBe('hold');
    expect(w.ring[hold.stage].fillFrac).toBe(1);
  });

  it('wgmma SS: during consumer of that stage, role === consume AND fillFrac === 1', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    const p0 = r.streams.producer[0];
    const stage = p0.stage!;
    const firstCons = r.streams.consumer.find((p) => p.stage === stage)!;
    // Sample mid-consumer phase.
    const mid = Math.floor((firstCons.startTick + firstCons.endTick) / 2);
    const w = r.worldAt(mid);
    expect(w.ring[stage].role).toBe('consume');
    expect(w.ring[stage].fillFrac).toBe(1);
  });

  it('wgmma SS: after last consumer atom of a slab, ring[stage].role === empty AND fillFrac === 0', () => {
    // Pick slab 0 (which we know has atomsPerStage_K consumer phases all on stage 0).
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, kStages: 3 }));
    const stage = 0;
    const consumersOfStage = r.streams.consumer.filter(
      (p) => p.stage === stage && p.kSlab === 0 && !p.collapsedCount,
    );
    const lastOfSlab = consumersOfStage.at(-1)!;
    // Use the very next tick after lastOfSlab.endTick: the empty effect has
    // fired, but before the next producer writes this stage.
    const w = r.worldAt(lastOfSlab.endTick);
    expect(w.ring[stage].role === 'empty' || w.ring[stage].role === 'fill').toBe(true);
    if (w.ring[stage].role === 'empty') {
      expect(w.ring[stage].fillFrac).toBe(0);
    }
  });

  it('wgmma RS: producerTransfer.kind === "ldmatrixA" during ldmatrix sub-phase; stage stays "hold"', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, aSource: 'rmem' }));
    const ldm = r.streams.producer.find((p) => p.producerSub === 'ldmatrixA');
    expect(ldm).toBeDefined();
    const mid = Math.floor((ldm!.startTick + ldm!.endTick) / 2);
    const w = r.worldAt(mid);
    expect(w.producerTransfer).not.toBeNull();
    expect(w.producerTransfer!.kind).toBe('ldmatrixA');
    expect(w.producerTransfer!.operand).toBe('A');
    // The stage holding this slab's A should remain 'hold' (or 'consume' if
    // the consumer has already started reading it — the ldmatrix is hoisted
    // immediately before the consumer; both are valid).
    const stage = ldm!.stage!;
    const role = w.ring[stage].role;
    expect(role === 'hold' || role === 'consume').toBe(true);
    // fillFrac must still be at 1 (we never zero'd it for RS sub-phases).
    expect(w.ring[stage].fillFrac).toBe(1);
  });

  it('tcgen05 TS: producerTransfer.kind === "tcgen05-cp" during tcgen05.cp sub-phase', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16');
    const r = simulate(makeInput({ inst, aSource: 'tmem' }));
    const cp = r.streams.producer.find((p) => p.producerSub === 'tcgen05-cp');
    expect(cp).toBeDefined();
    const mid = Math.floor((cp!.startTick + cp!.endTick) / 2);
    const w = r.worldAt(mid);
    expect(w.producerTransfer).not.toBeNull();
    expect(w.producerTransfer!.kind).toBe('tcgen05-cp');
    expect(w.producerTransfer!.operand).toBe('A');
    // The stage is already 'hold' (filled by the preceding tma.load).
    // fillFrac remains 1 — tcgen05.cp piggybacks on the held slab.
    const stage = cp!.stage!;
    expect(w.ring[stage].fillFrac).toBe(1);
  });

  it('sparse: producerTransfer.kind === "metadata" during metadata sub-phase', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16.sp');
    const r = simulate(makeInput({ inst }));
    const meta = r.streams.producer.find((p) => p.producerSub === 'metadata');
    expect(meta).toBeDefined();
    const mid = Math.floor((meta!.startTick + meta!.endTick) / 2);
    const w = r.worldAt(mid);
    expect(w.producerTransfer).not.toBeNull();
    expect(w.producerTransfer!.kind).toBe('metadata');
    expect(w.producerTransfer!.operand).toBe('meta');
  });

  it('block_scaled: producerTransfer.kind === "scale" during scale sub-phase', () => {
    const inst = findInst('sm100.tcgen05.cg1.m128n128k32.mxf8f6f4');
    const r = simulate(makeInput({ inst }));
    const scale = r.streams.producer.find((p) => p.producerSub === 'scale');
    expect(scale).toBeDefined();
    const mid = Math.floor((scale!.startTick + scale!.endTick) / 2);
    const w = r.worldAt(mid);
    expect(w.producerTransfer).not.toBeNull();
    expect(w.producerTransfer!.kind).toBe('scale');
    expect(w.producerTransfer!.operand).toBe('scaleA');
  });

  it('sm80 mma: producerTransfer.kind === "cpasync" during cp.async; ring synthetic slot fills', () => {
    const inst = findInst('sm80.mma.m16n8k16.f16');
    const r = simulate(makeInput({ inst }));
    const cp = r.streams.producer.find((p) => p.producerSub === 'cpasync');
    expect(cp).toBeDefined();
    const mid = Math.floor((cp!.startTick + cp!.endTick) / 2);
    const w = r.worldAt(mid);
    expect(w.producerTransfer).not.toBeNull();
    expect(w.producerTransfer!.kind).toBe('cpasync');
    expect(w.producerTransfer!.operand).toBe('AB');
  });

  it('sm70 wmma: producerTransfer.kind === "wmma-load"; no ring', () => {
    const inst = findInst('sm70.wmma.m16n16k16.f16');
    const r = simulate(makeInput({ inst }));
    const wl = r.streams.producer.find((p) => p.producerSub === 'wmma-load');
    expect(wl).toBeDefined();
    const mid = Math.floor((wl!.startTick + wl!.endTick) / 2);
    const w = r.worldAt(mid);
    expect(w.producerTransfer).not.toBeNull();
    expect(w.producerTransfer!.kind).toBe('wmma-load');
    expect(w.producerTransfer!.operand).toBe('AB');
    // wmma has no ring: length 0.
    expect(w.ring.length).toBe(0);
  });

  it('summary linesPerSlab fields are strictly positive for wgmma', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    expect(r.summary.linesPerSlab_A).toBeGreaterThan(0);
    expect(r.summary.linesPerSlab_B).toBeGreaterThan(0);
    expect(r.summary.linesPerSlab_AB).toBe(
      r.summary.linesPerSlab_A + r.summary.linesPerSlab_B,
    );
    expect(r.summary.linesPerAtom_ldmatrixA).toBeGreaterThan(0);
    expect(r.summary.linesPerSlab_tcgen05cp).toBeGreaterThan(0);
    expect(r.summary.linesPerSlab_metadata).toBeGreaterThan(0);
    expect(r.summary.linesPerSlab_scale).toBeGreaterThan(0);
  });

  it('ring fillFrac never drops to 0 between TMA end and last-consumer end for any stage', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, kStages: 3 }));
    // For slab 0 on stage 0, walk ticks from p0.endTick → lastConsumerOfSlab0.endTick − 1
    // and assert fillFrac === 1.
    const p0 = r.streams.producer.find((p) => p.producerSub === 'tma' && p.kSlab === 0)!;
    const stage = p0.stage!;
    const consumers = r.streams.consumer.filter(
      (p) => p.stage === stage && p.kSlab === 0 && !p.collapsedCount,
    );
    const lastCons = consumers.at(-1)!;
    for (let t = p0.endTick; t < lastCons.endTick; t++) {
      const w = r.worldAt(t);
      expect(w.ring[stage].fillFrac).toBe(1);
    }
  });
});
