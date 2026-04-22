import { activeSwizzleA, focusedOffset, inst, spec } from '../state';
import { apply, bankOfByte } from '../swizzle';
import { tileDimsFor } from '../tile_dims';

// Width of the displayed bitfield. 16 is enough for offsets up to 64 KiB; any
// larger tile needs the upper bits rendered too. We clamp the offset to 16
// low bits for readability and note when truncation happens.
const WIDTH = 16;

type Kind = 'preserved' | 'xor-src' | 'xor-dst' | 'untouched';

function classifyBits(B: number, M: number, S: number): Kind[] {
  const out: Kind[] = Array.from({ length: WIDTH }, () => 'untouched');
  for (let i = 0; i < M && i < WIDTH; i++) out[i] = 'preserved';
  if (B === 0) return out;
  const yyyShift = M + Math.max(0, S);
  for (let i = 0; i < B; i++) {
    const bit = yyyShift + i;
    if (bit < WIDTH) out[bit] = 'xor-src';
  }
  for (let i = 0; i < B; i++) {
    const bit = yyyShift + i - S;
    if (bit >= 0 && bit < WIDTH) {
      if (out[bit] !== 'xor-src') out[bit] = 'xor-dst';
    }
  }
  return out;
}

function bitAt(n: number, i: number): 0 | 1 {
  return ((n >>> i) & 1) as 0 | 1;
}

function hex(n: number, digits = 4): string {
  return `0x${(n >>> 0).toString(16).padStart(digits, '0')}`;
}

export function BitfieldPanel() {
  const s = spec.value;
  const i = inst.value;
  // Bitfield reports the current swizzle as it applies to operand A (the
  // usual "from my perspective" view). `activeSwizzleA` picks the per-dtype
  // byte-level M so switching dtype moves the preserved bits.
  const aDims = tileDimsFor(i, 'A', s.majorA);
  const sw = activeSwizzleA.value;
  const offRaw = focusedOffset.value;
  const kinds = classifyBits(sw.B, sw.M, sw.S);

  // Pre-compute the XOR mask using the same formula as swizzle.ts.
  const bitMask = sw.B === 0 ? 0 : (1 << sw.B) - 1;
  const yyyMask = (bitMask << (sw.M + Math.max(0, sw.S))) >>> 0;

  // Example offset when nothing is focused — use a value that highlights the
  // swizzle behavior for the current (B,M,S) triple, so the legend is live.
  const example = yyyMask | (1 << sw.M);
  const off = offRaw ?? example;
  const focused = offRaw !== null;

  const picked = ((off & yyyMask) >>> sw.S) >>> 0;
  const phys = apply(sw, off);
  const bank = bankOfByte(phys);
  const truncated = off >= 1 << WIDTH;

  const row = (n: number) =>
    Array.from({ length: WIDTH }, (_, i) => {
      const bitIdx = WIDTH - 1 - i;
      const bit = bitAt(n, bitIdx);
      const k = kinds[bitIdx];
      return (
        <span class={`bit bit--${k} ${bit ? 'bit--on' : ''}`} title={`bit ${bitIdx}`}>
          {bit}
        </span>
      );
    });

  return (
    <div class="panel bitfield-panel">
      <h3>
        Bitfield{' '}
        <small>
          A · {aDims.dtypeLabel} · M={sw.M} (preserve {1 << sw.M} B) ·{' '}
          {focused ? <>offset {hex(off & 0xffff)}</> : <>hover SMEM to focus</>}
        </small>
      </h3>
      <p class="bitfield__formula">
        <code>apply(o) = o ⊕ ((o &amp; yyy_msk) ≫ S)</code>
      </p>
      <div class="bitfield">
        <div class="bitfield__row">
          <span class="bitfield__label">logical</span>
          <div class="bitfield__bits">{row(off & 0xffff)}</div>
          <span class="bitfield__hex">{hex(off & 0xffff)}</span>
        </div>
        <div class="bitfield__row bitfield__row--ops">
          <span class="bitfield__label">⊕ picked</span>
          <div class="bitfield__bits">{row(picked & 0xffff)}</div>
          <span class="bitfield__hex">{hex(picked & 0xffff)}</span>
        </div>
        <div class="bitfield__row bitfield__row--result">
          <span class="bitfield__label">physical</span>
          <div class="bitfield__bits">{row(phys & 0xffff)}</div>
          <span class="bitfield__hex">{hex(phys & 0xffff)}</span>
        </div>
      </div>
      <div class="bitfield__legend">
        <span class="bit bit--preserved bit--on">P</span> preserved (bit &lt; M)
        <span class="bit bit--xor-src bit--on">Y</span> yyy (source)
        <span class="bit bit--xor-dst bit--on">Z</span> zzz (dest)
      </div>
      <p class="bitfield__foot">
        bank = (phys ≫ 2) &amp; 31 = <code>{bank}</code>
        {!focused && <span class="bitfield__live-hint"> · showing example, hover to focus real offset</span>}
        {truncated && <span class="bitfield__trunc"> · showing low 16 bits of 0x{off.toString(16)}</span>}
      </p>
    </div>
  );
}
