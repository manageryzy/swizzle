import { describe, expect, it } from 'vitest';
import { ringState } from './pipeline_state';

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
