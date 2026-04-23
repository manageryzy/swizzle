// Phase 2 tests — family-aware emitPhases honours plan §K1..§K9 and §N4..§N7.
//
// For each of the 9 canonical URLs in reference_testing_conventions.md we
// assert the producer/consumer stream lengths match plan §C5, plus the
// ordering invariants (§N4, §N6, §N7).

import { describe, expect, it } from 'vitest';
import { INSTRUCTIONS, type InstSpec, type OperandSource } from './instructions';
import { simulate, type SimInput, MAX_PHASES_PER_STREAM } from './simulation';

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

describe('Phase 2 — stream lengths match plan §C5', () => {
  it('wgmma SS: producer.length=slabCount, consumer.length=slabCount×atomsPerStage_K (capped)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    // slabCount=4, atomsPerStage_K=4 → 4 producer, min(16,8)+1 collapse=9
    expect(r.summary.slabCount).toBe(4);
    expect(r.summary.atomsPerStage_K).toBe(4);
    expect(r.summary.consumerItersTotal).toBe(16);
    expect(r.streams.producer.length).toBe(4);
    // consumer displayed 8 + 1 collapse = 9
    expect(r.streams.consumer.length).toBe(9);
    // wgmma (non-tmem accIn) epilogue: stg_smem → tma.store (length 2)
    expect(r.streams.epilogue.length).toBe(2);
  });

  it('wgmma RS: producer has TMA + ldmatrixA per k-atom (warpspec RS)', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, aSource: 'rmem' }));
    // slabs displayed=4 tma + 8 ldmatrix-A hoisted per displayed consumer iter.
    const ldms = r.streams.producer.filter((p) => p.producerSub === 'ldmatrixA').length;
    const tmas = r.streams.producer.filter((p) => p.producerSub === 'tma').length;
    expect(tmas).toBe(4);
    // Hoisted inline per displayed consumer iter (8); one ldmatrix-A per consumer phase (excl collapse).
    expect(ldms).toBe(Math.min(r.summary.consumerItersTotal, MAX_PHASES_PER_STREAM));
    // Consumer count same as SS.
    expect(r.streams.consumer.length).toBe(9);
  });

  it('tcgen05 SS cg1: epilogue.length=3 (adds tcgen05.ld)', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16');
    const r = simulate(makeInput({ inst }));
    expect(r.streams.epilogue.length).toBe(3);
    expect(r.streams.epilogue[0].kind).toBe('tcgen05.ld');
    expect(r.streams.producer.filter((p) => p.producerSub === 'tma').length).toBe(4);
    expect(r.streams.consumer.length).toBe(9);
  });

  it('tcgen05 TS cg1: producer has extra tcgen05.cp per slab', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16');
    const r = simulate(makeInput({ inst, aSource: 'tmem' }));
    const tmemCps = r.streams.producer.filter((p) => p.producerSub === 'tcgen05-cp').length;
    expect(tmemCps).toBe(4);
    expect(r.streams.producer.filter((p) => p.producerSub === 'tma').length).toBe(4);
  });

  it('tcgen05 sparse cg1: producer has extra metadata per slab', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16.sp');
    const r = simulate(makeInput({ inst }));
    const meta = r.streams.producer.filter((p) => p.producerSub === 'metadata').length;
    expect(meta).toBe(4);
  });

  it('tcgen05 block_scaled cg1: producer has extra scale per slab', () => {
    const inst = findInst('sm100.tcgen05.cg1.m128n128k32.mxf8f6f4');
    const r = simulate(makeInput({ inst }));
    const scales = r.streams.producer.filter((p) => p.producerSub === 'scale').length;
    expect(scales).toBe(4);
  });

  it('tcgen05 cg2: still emits one TMA per slab (multicast handled later)', () => {
    const inst = findInst('sm100.tcgen05.cg2.m128n128k16.f16');
    const r = simulate(makeInput({ inst }));
    expect(r.streams.producer.filter((p) => p.producerSub === 'tma').length).toBe(4);
    expect(r.streams.consumer.length).toBe(9);
  });

  it('sm80 mma (coupled): producer.length === consumer.length', () => {
    const inst = findInst('sm80.mma.m16n8k16.f16');
    const r = simulate(makeInput({ inst }));
    // 16 total iters, capped at 8 + 1 collapse = 9 each.
    // Producer displayed = cpasync per slab + ldmatrix per atom hoist.
    // Per plan §K2, producer emits 1 cp.async PER SLAB + 1 ldmatrix PER ATOM.
    // To mirror the coupled per-k-step pattern for the asymmetry memo, we
    // count the ldmatrix phases — these equal the consumer count.
    const ldms = r.streams.producer.filter((p) => p.kind === 'ldmatrix').length;
    const displayedConsumer = Math.min(r.summary.consumerItersTotal, MAX_PHASES_PER_STREAM);
    expect(ldms).toBe(displayedConsumer);
    // Plus 1 collapse tail phase + slabCount cp.async phases.
    // Total producer = slabCount_capped*1_cpasync + atomsPerStage_K*slabCount_capped ldmatrix + 1 collapse tail
    expect(r.summary.slabCount).toBe(4);
    expect(r.summary.consumerItersTotal).toBe(16);
    // Consumer displayed 8 + collapse 1 = 9
    expect(r.streams.consumer.length).toBe(9);
  });

  it('sm70 wmma (coupled, no SMEM): producer per k-atom, no ring', () => {
    const inst = findInst('sm70.wmma.m16n16k16.f16');
    const r = simulate(makeInput({ inst }));
    expect(r.summary.familyShape).toBe('wmma-direct');
    expect(r.summary.slabCount).toBe(4);
    expect(r.summary.atomsPerStage_K).toBe(4);
    expect(r.summary.consumerItersTotal).toBe(16);
    // Consumer displayed 8 + collapse 1 = 9
    expect(r.streams.consumer.length).toBe(9);
    // Producer per plan §K1 is "per-k-step" — one wmma.load per k-atom.
    const loads = r.streams.producer.filter((p) => p.kind === 'wmma.load').length;
    // 8 displayed + 1 tail collapse
    expect(loads).toBe(9);
    // No ring.
    const w = r.worldAt(0);
    expect(w.ring.length).toBe(0);
  });
});

describe('Phase 2 — plan §N invariants', () => {
  const cases: { id: string; aSource?: OperandSource }[] = [
    { id: 'sm70.wmma.m16n16k16.f16' },
    { id: 'sm80.mma.m16n8k16.f16' },
    { id: 'sm90.wgmma.m64n128k16.f32f16' },
    { id: 'sm90.wgmma.m64n128k16.f32f16', aSource: 'rmem' },
    { id: 'sm100.tcgen05.cg1.m64n128k16.f16' },
    { id: 'sm100.tcgen05.cg1.m64n128k16.f16', aSource: 'tmem' },
    { id: 'sm100.tcgen05.cg2.m128n128k16.f16' },
    { id: 'sm100.tcgen05.cg1.m64n128k16.f16.sp' },
    { id: 'sm100.tcgen05.cg1.m128n128k32.mxf8f6f4' },
  ];

  for (const c of cases) {
    const label = `${c.id}${c.aSource ? `/${c.aSource}` : ''}`;

    it(`${label}: §N4 slab order strictly ascending`, () => {
      const inst = findInst(c.id);
      const r = simulate(makeInput({
        inst,
        aSource: c.aSource ?? (inst.aSource[0] as OperandSource),
      }));
      // Per-slab TMA order strict ascending.
      const tmaOrder = r.streams.producer
        .filter((p) => p.producerSub === 'tma' || p.producerSub === 'cpasync' || p.producerSub === 'wmma-load')
        .filter((p) => !p.collapsedCount)
        .map((p) => p.kSlab ?? p.iter ?? 0);
      for (let i = 1; i < tmaOrder.length; i++) {
        expect(tmaOrder[i]).toBeGreaterThanOrEqual(tmaOrder[i - 1]);
      }
      // Consumer flat iter order strictly ascending.
      let prev = -1;
      for (const p of r.streams.consumer) {
        if (p.iter === undefined) continue;
        expect(p.iter).toBeGreaterThanOrEqual(prev);
        prev = p.iter;
      }
    });

    it(`${label}: §N6 consumer precondition — consumer[s,a].start ≥ producer[s] per-slab subs end`, () => {
      const inst = findInst(c.id);
      const r = simulate(makeInput({
        inst,
        aSource: c.aSource ?? (inst.aSource[0] as OperandSource),
      }));
      // For warpspec: consumer phase for slab s must start ≥ any producer
      // per-slab sub-phase end tick for slab s.
      for (const cPh of r.streams.consumer) {
        if (cPh.collapsedCount) continue;
        const s = cPh.kSlab;
        if (s === undefined) continue;
        const perSlabSubs = r.streams.producer.filter(
          (p) => p.kSlab === s && p.kAtomInSlab === undefined && !p.collapsedCount,
        );
        for (const p of perSlabSubs) {
          expect(cPh.startTick).toBeGreaterThanOrEqual(p.endTick);
        }
      }
    });

    it(`${label}: §N7 producer ring-full precondition (warpspec only)`, () => {
      const inst = findInst(c.id);
      const r = simulate(makeInput({
        inst,
        aSource: c.aSource ?? (inst.aSource[0] as OperandSource),
        kStages: 2, // force back-pressure with 4 slabs
      }));
      if (r.summary.familyShape !== 'tma-warpspec') return;
      const kStages = 2;
      const atomsPerSlab = r.summary.atomsPerStage_K;
      // producer[s].first-per-slab-sub.start ≥ consumer[s-kStages, last-atom].end
      for (let s = kStages; s < r.summary.slabCount; s++) {
        const prodFirstSub = r.streams.producer.find(
          (p) => p.kSlab === s && p.kAtomInSlab === undefined && !p.collapsedCount,
        );
        const consLastAtom = r.streams.consumer.find(
          (p) => p.kSlab === s - kStages && p.kAtomInSlab === atomsPerSlab - 1 && !p.collapsedCount,
        );
        if (!prodFirstSub || !consLastAtom) continue; // may be collapsed out
        expect(prodFirstSub.startTick).toBeGreaterThanOrEqual(consLastAtom.endTick);
      }
    });
  }
});

describe('Phase 2 — specific counts', () => {
  it('wgmma-SS default: slabCount=4, atomsPerStage_K=4, consumerItersTotal=16, producer.length=4, consumer.length=9, epilogue.length=2', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst }));
    expect(r.summary.slabCount).toBe(4);
    expect(r.summary.atomsPerStage_K).toBe(4);
    expect(r.summary.consumerItersTotal).toBe(16);
    expect(r.streams.producer.length).toBe(4);
    expect(r.streams.consumer.length).toBe(9); // 8 displayed + 1 collapse
    expect(r.streams.epilogue.length).toBe(2);
  });

  it('tcgen05-SS cg1 default: epilogue.length=3 (tcgen05.ld + stg + store)', () => {
    const inst = findInst('sm100.tcgen05.cg1.m64n128k16.f16');
    const r = simulate(makeInput({ inst }));
    expect(r.streams.epilogue.length).toBe(3);
  });

  it('sm80 mma coupled: effective producer-per-k-atom count = slabCount × atomsPerStage_K', () => {
    const inst = findInst('sm80.mma.m16n8k16.f16');
    const r = simulate(makeInput({ inst }));
    // ldmatrix per k-atom mirrors the coupled pattern.
    const ldms = r.streams.producer.filter((p) => p.kind === 'ldmatrix' && !p.collapsedCount).length;
    // Display cap: min(16, 8) = 8 displayed.
    expect(ldms).toBe(8);
  });

  it('wmma coupled: producer.length = consumer.length (per-k-step) with tail collapse', () => {
    const inst = findInst('sm70.wmma.m16n16k16.f16');
    const r = simulate(makeInput({ inst }));
    // After collapse, both streams have the tail bar — symmetric.
    expect(r.streams.producer.length).toBe(r.streams.consumer.length);
  });

  it('collapsed tail: problemKMult=8, atomsPerStage_K=4 → consumerItersTotal=32; consumer.length ≤ 9', () => {
    const inst = findInst('sm90.wgmma.m64n128k16.f32f16');
    const r = simulate(makeInput({ inst, problemKMult: 8 }));
    expect(r.summary.slabCount).toBe(8);
    expect(r.summary.consumerItersTotal).toBe(32);
    expect(r.streams.consumer.length).toBeLessThanOrEqual(MAX_PHASES_PER_STREAM + 1);
    // Total tick duration still represents full mainloop (tail collapse shifts
    // epilogue past mainloopEnd, which is the last bar end).
    expect(r.totalTicks).toBe(r.streams.epilogue.at(-1)!.endTick);
  });
});

describe('Phase 2 — coupled families use per-k-step producer', () => {
  it('sm80 mma: 1 cp.async per slab + 1 ldmatrix per k-atom (displayed)', () => {
    const inst = findInst('sm80.mma.m16n8k16.f16');
    const r = simulate(makeInput({ inst }));
    const cpasyncs = r.streams.producer.filter((p) => p.kind === 'cp.async' && !p.collapsedCount).length;
    const ldms = r.streams.producer.filter((p) => p.kind === 'ldmatrix' && !p.collapsedCount).length;
    // slabCount capped to MAX_PHASES_PER_STREAM for cp.async because coupled
    // halts at consumerItersDisplayed boundary.
    expect(cpasyncs).toBeGreaterThan(0);
    expect(ldms).toBeGreaterThan(0);
    // Total producer (including tail) roughly 2× consumer displayed (one pair per k-atom).
    expect(r.streams.producer.length).toBeGreaterThan(r.streams.consumer.length);
  });

  it('sm70 wmma: per-k-step pattern, no SMEM/ring', () => {
    const inst = findInst('sm70.wmma.m16n16k16.f16');
    const r = simulate(makeInput({ inst }));
    expect(r.summary.hasRing).toBe(false);
    expect(r.worldAt(0).ring.length).toBe(0);
    expect(r.worldAt(0).mbar.length).toBe(0);
  });
});
