import { ConfigBar } from './panels/ConfigBar';
import { SmemPanel } from './panels/SmemPanel';
import { TmemPanel } from './panels/TmemPanel';
import { RmemPanel } from './panels/RmemPanel';
import { Timeline } from './panels/Timeline';
import { ConflictMeter } from './panels/ConflictMeter';
import { BitfieldPanel } from './panels/BitfieldPanel';
import { CutlassTokens } from './panels/CutlassTokens';
import { RfBanksPanel } from './panels/RfBanksPanel';
import { SmemBudgetPanel } from './panels/SmemBudgetPanel';
import { buildDescriptor } from './descriptor';
import { activeSwizzle, inst, spec } from './state';

export function App() {
  const i = inst.value;
  const sw = activeSwizzle.value;
  const s = spec.value;

  const descArch = i.arch === 'sm100' ? 'sm100' : 'sm90';
  const desc = buildDescriptor({
    arch: descArch,
    startByte: 0x0100,
    leadingByteOffset: 16,
    strideByteOffset: 64,
    swizzle: sw,
  });

  return (
    <main>
      <div class="sticky-top">
        <header>
          <h1>swizzle matmul</h1>
          <small>cute · wmma · mma · wgmma · tcgen05</small>
          <span class="header__badge">
            {i.arch} · <code>{i.family}</code> · m{i.M}n{i.N}k{i.K}
            {i.ctaGroup ? ` · cg${i.ctaGroup}` : ''} · {s.swizzle}
          </span>
        </header>
        <ConfigBar />
        <Timeline />
      </div>

      <div class="body-grid">
        <div class="main-column">
          <section class="section section--memory">
            <header class="section__header">
              <span class="section__rail" />
              <h2>Memory hierarchy</h2>
              <p>physical layout of SMEM, TMEM, and the register file as the kernel moves data GMEM → SMEM → LRF → TMEM → GMEM.</p>
            </header>
            <SmemBudgetPanel />
            <SmemPanel />
            <div class="panels-row">
              <TmemPanel />
              <RmemPanel />
            </div>
            <RfBanksPanel />
          </section>

          <section class="section section--reference">
            <header class="section__header">
              <span class="section__rail section__rail--reference" />
              <h2>Reference</h2>
              <p>CUTLASS tokens and the 64-bit matrix descriptor for the current config. Click to expand.</p>
            </header>
            <details>
              <summary>CUTLASS tokens (arch tag, atom names, kernel schedule)</summary>
              <CutlassTokens />
            </details>
            <details>
              <summary>
                {descArch === 'sm90' ? 'GmmaDescriptor' : 'SmemDescriptor'} — layout_type ={' '}
                <code>{desc.layoutTypeName}</code>
              </summary>
              <div class="panel desc">
                <table>
                  <thead>
                    <tr>
                      <th>field</th>
                      <th>bits</th>
                      <th>value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {desc.fields.map((f) => (
                      <tr>
                        <td><code>{f.name}</code></td>
                        <td>[{f.bits[0]},{f.bits[1]})</td>
                        <td><code>0x{f.value.toString(16)}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p>
                  raw = <code>0x{desc.raw.toString(16).padStart(16, '0')}</code> · arch =
                  <code> {desc.arch}</code>
                </p>
              </div>
            </details>
          </section>
        </div>

        <aside class="swizzle-sidebar">
          <header class="section__header section__header--tight">
            <span class="section__rail section__rail--analysis" />
            <h2>Swizzle analysis</h2>
          </header>
          <ConflictMeter />
          <BitfieldPanel />
        </aside>
      </div>
    </main>
  );
}
