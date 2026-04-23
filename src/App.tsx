import { useState } from 'preact/hooks';
import { ConfigBar } from './panels/ConfigBar';
import { SmemPanel } from './panels/SmemPanel';
import { TileHierarchyPanel } from './panels/TileHierarchyPanel';
import { MemFlowPanel } from './panels/MemFlowPanel';
import { GmemPanel } from './panels/GmemPanel';
import { TmemPanel } from './panels/TmemPanel';
import { SimtRegPanel } from './panels/SimtRegPanel';
import { Timeline } from './panels/Timeline';
import { ConflictMeter } from './panels/ConflictMeter';
import { ConflictMatrix } from './panels/ConflictMatrix';
import { CutlassTokens } from './panels/CutlassTokens';
import { SmemBudgetPanel } from './panels/SmemBudgetPanel';
import { buildDescriptor } from './descriptor';
import {
  activeSwizzle,
  inst,
  spec,
  warpSel,
  warpsInGroup,
  laneSel,
  densityMode,
} from './state';

function SimtStrip() {
  const warps = warpsInGroup.value;
  const w = warpSel.value;
  const l = laneSel.value;
  const d = densityMode.value;
  const [showHelp, setShowHelp] = useState(false);
  return (
    <>
      <div class="simt-strip">
        <span class="simt-strip__label">warpgroup</span>
        <div class="simt-strip__warps">
          {Array.from({ length: warps }, (_, i) => (
            <button
              class={`simt-strip__warp ${i === w ? 'is-active' : ''}`}
              onClick={() => (warpSel.value = i)}
              title={`warp ${i} (32 lanes)`}
            >
              W{i}
            </button>
          ))}
        </div>
        <span class="simt-strip__sep">·</span>
        <span class="simt-strip__label">lane</span>
        <button
          class={`simt-strip__lane ${l == null ? 'is-null' : ''}`}
          onClick={() => (laneSel.value = l == null ? 0 : null)}
          title="cycle lane focus (also: key l)"
        >
          {l == null ? 'none' : `L${l.toString().padStart(2, '0')}`}
        </button>
        <div class="simt-strip__density">
          <button class={d === 'compact' ? 'is-active' : ''} onClick={() => (densityMode.value = 'compact')}>compact</button>
          <button class={d === 'detail' ? 'is-active' : ''} onClick={() => (densityMode.value = 'detail')}>detail</button>
        </div>
        <span class="simt-strip__help" title="keyboard shortcuts" onClick={() => setShowHelp(true)}>?</span>
      </div>
      {showHelp && <KbdOverlay onClose={() => setShowHelp(false)} />}
    </>
  );
}

function KbdOverlay({ onClose }: { onClose: () => void }) {
  return (
    <div class="kbd-overlay" onClick={onClose}>
      <div class="kbd-overlay__inner" onClick={(e: Event) => e.stopPropagation()}>
        <h3>keyboard</h3>
        <dl>
          <dt>space</dt><dd>play / pause</dd>
          <dt>← →</dt><dd>±1 tick</dd>
          <dt>shift + ← →</dt><dd>prev / next phase</dd>
          <dt>[ ]</dt><dd>prev / next replay cycle</dd>
          <dt>l</dt><dd>cycle lane focus</dd>
          <dt>w</dt><dd>cycle warp focus</dd>
          <dt>d</dt><dd>toggle density</dd>
        </dl>
        <button class="kbd-overlay__close" onClick={onClose}>close</button>
      </div>
    </div>
  );
}

export function App() {
  const i = inst.value;
  const sw = activeSwizzle.value;
  const s = spec.value;
  const d = densityMode.value;

  const descArch = i.arch === 'sm100' ? 'sm100' : 'sm90';
  const desc = buildDescriptor({
    arch: descArch,
    startByte: 0x0100,
    leadingByteOffset: 16,
    strideByteOffset: 64,
    swizzle: sw,
  });

  return (
    <main data-density={d}>
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
        <SimtStrip />
      </div>

      <Timeline />

      <div class="body-grid">
        <div class="main-column">
          <section class="section section--memory">
            <header class="section__header">
              <span class="section__rail" />
              <h2>Memory hierarchy</h2>
              <p>physical layout of SMEM, TMEM, and the register file as the kernel moves data GMEM → SMEM → LRF → TMEM → GMEM.</p>
            </header>
            <TileHierarchyPanel />
            <MemFlowPanel />
            {/* GmemPanel, SmemBudgetPanel, SmemPanel, ConflictMatrix: all
                hidden for wmma because that path does not stage operands
                through SMEM (no swizzle, no bank mapping, no ring buffer).
                For sm_80 mma and sm_90/100 the full SMEM swizzle story
                applies. */}
            {i.family !== 'wmma' && (
              <>
                <GmemPanel />
                <SmemBudgetPanel />
                <SmemPanel />
                <ConflictMatrix />
              </>
            )}
            <TmemPanel />
            <SimtRegPanel />
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

        {/* Swizzle analysis sidebar is only meaningful for families that
            stage operands through swizzled SMEM. wmma reads straight from
            gmem/shared into the fragment, so there's nothing to rank. */}
        {i.family !== 'wmma' && (
          <aside class="swizzle-sidebar">
            <header class="section__header section__header--tight">
              <span class="section__rail section__rail--analysis" />
              <h2>Swizzle analysis</h2>
            </header>
            <ConflictMeter />
          </aside>
        )}
      </div>
    </main>
  );
}
