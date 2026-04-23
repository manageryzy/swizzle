import { describe, it, expect } from 'vitest';
import { INSTRUCTIONS } from './instructions';
import { simulate, type SimInput } from './simulation';

const canonical = [
  { id: 'sm70.wmma.m16n16k16.f16' },
  { id: 'sm80.mma.m16n8k16.f16' },
  { id: 'sm90.wgmma.m64n128k16.f32f16' },
  { id: 'sm90.wgmma.m64n128k16.f32f16', aSource: 'rmem' as const },
  { id: 'sm100.tcgen05.cg1.m64n128k16.f16' },
  { id: 'sm100.tcgen05.cg1.m64n128k16.f16', aSource: 'tmem' as const },
  { id: 'sm100.tcgen05.cg2.m128n128k16.f16' },
  { id: 'sm100.tcgen05.cg1.m64n128k16.f16.sp' },
  { id: 'sm100.tcgen05.cg1.m128n128k32.mxf8f6f4' },
];

describe('canonical URL smoke test — simulate runs without error', () => {
  for (const c of canonical) {
    it(`${c.id}${c.aSource ? ` (aSource=${c.aSource})` : ''}`, () => {
      const inst = INSTRUCTIONS.find((x) => x.id === c.id);
      expect(inst, `no InstSpec for ${c.id}`).toBeDefined();
      if (!inst) return;
      const aSource = c.aSource ?? inst.aSource[0];
      const input: SimInput = {
        inst,
        majorA: 'K',
        majorB: 'K',
        swizzle: '128B',
        dtypeA: inst.aDtypes[0],
        dtypeB: inst.bDtypes[0],
        accDtype: inst.accDtypes[0],
        aSource,
        blkMMult: 1,
        blkNMult: 1,
        tileK: inst.K * 4,
        kStages: 3,
        problemMMult: 4,
        problemNMult: 4,
        problemKMult: 4,
      };
      const r = simulate(input);
      expect(r.totalTicks).toBeGreaterThan(0);
      expect(r.streams.consumer.length).toBeGreaterThan(0);
      expect(r.streams.producer.length).toBeGreaterThan(0);
      // Walk 5 ticks; must not throw.
      for (const f of [0, 0.25, 0.5, 0.75, 1.0]) {
        const t = Math.floor(r.totalTicks * f);
        const w = r.worldAt(t);
        expect(w.tick).toBe(t);
      }
    });
  }
});
