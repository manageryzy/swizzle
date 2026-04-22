// Parity test: TS port must reproduce verify/expected.json byte-for-byte
// (values, not whitespace). If this fails, the TS port has drifted from the
// C++ reference in verify/verify_standalone.cpp.
//
// Regenerate expected.json after intentional changes:
//   cd verify && g++ -std=c++17 -O0 verify_standalone.cpp -o verify_standalone && ./verify_standalone > expected.json

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { SWIZZLES, apply, type Swizzle } from './swizzle';
import { C_LAYOUTS } from './cute_mma_layouts';
import { layoutAt, sizeOf } from './cute_layout';

const expected = JSON.parse(
  readFileSync(resolve(__dirname, '../verify/expected.json'), 'utf8'),
) as {
  swizzle: Record<string, Record<string, string>>;
  clayout: Record<string, { M: number; threads: number; values_per_thread: number; thread0: [number, number][] }>;
};

const hex = (n: number) => `0x${n.toString(16).padStart(3, '0')}`;

function swizzleName(k: keyof typeof SWIZZLES): string {
  switch (k) {
    case 'none': return 'NO_SWIZZLE';
    case '32B': return 'SW32';
    case '64B': return 'SW64';
    case '128B': return 'SW128';
    case '128B.base32B': return 'SW128_BASE32B';
  }
}

describe('TS ↔ C++ parity (verify/expected.json)', () => {
  for (const [kind, sw] of Object.entries(SWIZZLES) as [keyof typeof SWIZZLES, Swizzle][]) {
    it(`swizzle ${kind} matches verify_standalone`, () => {
      const golden = expected.swizzle[swizzleName(kind)];
      expect(golden, `missing key in expected.json: ${kind}`).toBeDefined();
      for (const [offHex, physHex] of Object.entries(golden)) {
        const off = Number.parseInt(offHex, 16);
        expect(hex(apply(sw, off)), `off=${offHex}`).toBe(physHex);
      }
    });
  }

  const instFor: Record<string, string> = {
    SM80_16x8_Row: 'sm80.mma.m16n8k16.f16',
    // F4 split wgmma by acc dtype — the fp32-acc variant matches the cute
    // atom (`SM90_64xNxK_F32F16F16_SS`) referenced by verify_standalone.
    CLayout_64x8: 'sm90.wgmma.m64n8k16.f32f16',
    CLayout_64x128: 'sm90.wgmma.m64n128k16.f32f16',
  };

  for (const [clName, instId] of Object.entries(instFor)) {
    it(`CLayout ${clName} thread-0 matches verify_standalone`, () => {
      const golden = expected.clayout[clName];
      const c = C_LAYOUTS[instId];
      expect(c, `missing CLayout in TS: ${instId}`).toBeDefined();
      expect(c!.M).toBe(golden.M);
      expect(c!.threads).toBe(golden.threads);
      expect(c!.valuesPerThread).toBe(golden.values_per_thread);
      const total = sizeOf(c!.shape);
      const ours: [number, number][] = [];
      for (let v = 0; v < c!.valuesPerThread; v++) {
        const idx = 0 + v * c!.threads;
        if (idx >= total) break;
        const off = layoutAt(c!.shape, c!.stride, idx);
        ours.push([off % c!.M, Math.floor(off / c!.M)]);
      }
      expect(ours).toEqual(golden.thread0);
    });
  }
});
