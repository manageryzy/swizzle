import { describe, expect, it } from 'vitest';
import { emitTimeline, ringState } from './pipeline_state';
import { INSTRUCTIONS } from './instructions';

describe('pipeline ring state', () => {
  it('kStages=3 at iter 0: stage 0 consumes slice 0; stages 1,2 hold 1,2', () => {
    const s = ringState(0, 3);
    expect(s).toEqual([
      { stage: 0, slice: 0, role: 'consume' },
      { stage: 1, slice: 1, role: 'hold' },
      { stage: 2, slice: 2, role: 'fill' },
    ]);
  });

  it('kStages=3 at iter 1: consumer rotates to stage 1', () => {
    const s = ringState(1, 3);
    // producer head = 3, slot distribution: stage 0 holds slice 3, stage 1
    // consumes slice 1, stage 2 holds slice 2.
    expect(s[0]).toMatchObject({ slice: 3, role: 'fill' });
    expect(s[1]).toMatchObject({ slice: 1, role: 'consume' });
    expect(s[2]).toMatchObject({ slice: 2, role: 'hold' });
  });

  it('kStages=3 at iter 3: stage 0 reused for slice 3', () => {
    const s = ringState(3, 3);
    // head = 5. consumer = 0 (slice 3). producer = 2 (writing slice 5).
    expect(s[0]).toMatchObject({ slice: 3, role: 'consume' });
    expect(s[1]).toMatchObject({ slice: 4, role: 'hold' });
    expect(s[2]).toMatchObject({ slice: 5, role: 'fill' });
  });

  it('kStages=1: no pipelining, stage 0 is always both consumer and producer', () => {
    const s0 = ringState(0, 1);
    const s5 = ringState(5, 1);
    expect(s0[0].role).toBe('consume'); // consumer wins the label when both match
    expect(s0[0].slice).toBe(0);
    expect(s5[0].slice).toBe(5);
  });

  it('slice numbers advance +1 as iter advances', () => {
    // At the consumer stage, slice == iter.
    for (let iter = 0; iter < 10; iter++) {
      const s = ringState(iter, 4);
      const consuming = s.find((x) => x.role === 'consume')!;
      expect(consuming.slice).toBe(iter);
    }
  });
});

describe('emitTimeline', () => {
  const wgmma = INSTRUCTIONS.find((i) => i.id === 'sm90.wgmma.m64n128k16.f32f16')!;
  const mma80 = INSTRUCTIONS.find((i) => i.id === 'sm80.mma.m16n8k16.f16')!;

  it('warpspec: consumer iter 0 waits for producer iter 0 before starting', () => {
    const t = emitTimeline(wgmma, 4, 3, 'warpspec');
    // Consumer[0] starts at Producer[0].endTick (stage 0 primed).
    expect(t.consumer[0].startTick).toBeGreaterThanOrEqual(t.producer[0].endTick);
  });

  it('warpspec: producer and consumer overlap in time for some pair', () => {
    const t = emitTimeline(wgmma, 4, 3, 'warpspec');
    // Some producer and some consumer phase occupy the same tick range.
    let found = false;
    for (const p of t.producer) {
      for (const c of t.consumer) {
        const overlapStart = Math.max(p.startTick, c.startTick);
        const overlapEnd = Math.min(p.endTick, c.endTick);
        if (overlapEnd > overlapStart) {
          found = true;
          break;
        }
      }
      if (found) break;
    }
    expect(found).toBe(true);
  });

  it('warpspec: producer iter k waits for consumer iter (k - kStages) when ring is full', () => {
    // With kStages=2 and numIters=4, producer iter 2 cannot start until
    // consumer iter 0 finishes (freeing stage 0).
    const t = emitTimeline(wgmma, 4, 2, 'warpspec');
    const prodIdx2 = t.producer[2];
    const consIdx0 = t.consumer[0];
    expect(prodIdx2.startTick).toBeGreaterThanOrEqual(consIdx0.endTick);
  });

  it('coupled: producer and consumer strictly serial, no overlap', () => {
    const t = emitTimeline(mma80, 3, 2, 'coupled');
    // Every consumer phase starts at-or-after every producer phase of the same iter.
    for (let k = 0; k < t.consumer.length - 1; k++) {
      const prev = t.consumer[k];
      const next = t.consumer[k + 1];
      expect(next.startTick).toBeGreaterThanOrEqual(prev.endTick);
    }
    // No overlap: any pair of producer & consumer phases — producer iter k must
    // end before consumer iter k begins (in coupled regime).
    for (let k = 0; k < 3; k++) {
      const p = t.producer.find((ph) => ph.iter === k && (ph.kind === 'cp.async' || ph.kind === 'tma.load'));
      const c = t.consumer[k];
      expect(c.startTick).toBeGreaterThanOrEqual(p!.endTick);
    }
  });

  it('epilogue phases always come after the mainloop drains', () => {
    const t = emitTimeline(wgmma, 4, 3, 'warpspec');
    const mainloopEnd = Math.max(
      t.producer.at(-1)?.endTick ?? 0,
      t.consumer.at(-1)?.endTick ?? 0,
    );
    for (const ph of t.epilogue) {
      expect(ph.startTick).toBeGreaterThanOrEqual(mainloopEnd);
    }
  });

  it('totalTicks equals the last epilogue phase endTick', () => {
    const t = emitTimeline(wgmma, 6, 3, 'warpspec');
    expect(t.totalTicks).toBe(t.epilogue.at(-1)!.endTick);
  });
});
