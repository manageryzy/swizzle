import { useEffect, useRef, useState } from 'preact/hooks';
import { currentKStep, currentPhase, inst, phaseProgress } from '../state';
import {
  TMEM_DP,
  TMEM_COLS,
  TMEM_SUBPARTITIONS,
  TCGEN05_SHAPES,
  accFootprint,
  findShape,
} from '../tmem';
import { PTX_LAYOUTS, classifyLayout, peerMRange } from '../tcgen05_layouts';

const CELL = 2;
const ROWS_PX = TMEM_DP * (CELL + 1);

function subColor(sub: number): string {
  return `hsl(${(sub * 360) / TMEM_SUBPARTITIONS}, 55%, 35%)`;
}

export function TmemPanel() {
  const i = inst.value;
  const phase = currentPhase.value;
  const progress = phaseProgress.value;
  const kStep = currentKStep.value;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [shapeId, setShapeId] = useState<string>('16dp256b x4');
  const [peer, setPeer] = useState<0 | 1>(0);

  const usesTmem = i.accIn === 'tmem' || i.aSource.includes('tmem');
  const ctaGroup = (i.ctaGroup ?? 1) as 1 | 2;
  const ptxLayout = usesTmem
    ? classifyLayout(i.M, ctaGroup, { sparse: i.sparse, warpSpecialized: i.warpSpecialized })
    : null;
  const layoutInfo = ptxLayout ? PTX_LAYOUTS[ptxLayout] : null;
  // Prefer the authoritative DP count from the layout classification; fall
  // back to the numeric accFootprint when outside the PTX taxonomy.
  const baseFootprint = accFootprint(i.M, i.N, ctaGroup);
  const accDp = layoutInfo ? Math.min(TMEM_DP, layoutInfo.dpRows) : baseFootprint.dp;
  const accCols = baseFootprint.cols;
  const range = ptxLayout ? peerMRange(ptxLayout, peer, i.M) : null;

  const fill = (() => {
    if (phase?.kind === 'tcgen05.mma.step') {
      if (!kStep) return 0;
      return Math.min(1, (kStep.k + progress) / kStep.total);
    }
    return phase && (phase.kind === 'tcgen05.ld' || phase.kind === 'epilogue.tma.store') ? 1 : 0;
  })();

  // Show only first 128 columns (256B worth at 32b) so the panel fits without
  // horizontal scroll. M3+ can add virtual scrolling.
  const SHOWN_COLS = 128;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = SHOWN_COLS * (CELL + 1);
    const H = TMEM_DP * (CELL + 1);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // 1. Subpartition colouring — 32 DP per subpartition × cols.
    for (let dp = 0; dp < TMEM_DP; dp++) {
      const sub = Math.floor(dp / 32);
      ctx.fillStyle = subColor(sub);
      ctx.fillRect(0, dp * (CELL + 1), W, CELL + 1);
    }

    // 2. Accumulator footprint (bright overlay). Alpha ramps with `fill` so
    //    the footprint "fills up" as mma steps progress.
    if (usesTmem) {
      const alpha = 0.15 + 0.55 * fill;
      ctx.fillStyle = `rgba(255, 200, 120, ${alpha})`;
      const cw = Math.min(accCols, SHOWN_COLS) * (CELL + 1);
      const ch = accDp * (CELL + 1);
      ctx.fillRect(0, 0, cw, ch);
      // Leading edge of the fill.
      const edgeX = Math.floor(cw * fill);
      ctx.fillStyle = `rgba(255, 220, 140, ${0.4 + 0.4 * fill})`;
      if (fill > 0 && fill < 1) ctx.fillRect(edgeX - 2, 0, 3, ch);
      ctx.strokeStyle = '#ffc878';
      ctx.lineWidth = 1;
      ctx.strokeRect(0, 0, cw, ch);
    }

    // 3. tcgen05.ld/.st shape overlay — 16 DP × N bits × repeat.
    const shape = findShape(shapeId);
    if (shape && usesTmem) {
      const colsPerIssue = shape.bits / 32; // uint32 cells
      const totalCols = colsPerIssue * shape.repeat;
      // Position: top of accumulator footprint. 4 warp-groups of 16 DP split
      // the 128 DP; a single issue covers 16 DP.
      ctx.strokeStyle = '#7e9cd8';
      ctx.lineWidth = 2;
      ctx.strokeRect(0, 0, Math.min(totalCols, SHOWN_COLS) * (CELL + 1), shape.dp * (CELL + 1));
      ctx.fillStyle = 'rgba(126, 156, 216, 0.2)';
      ctx.fillRect(0, 0, Math.min(totalCols, SHOWN_COLS) * (CELL + 1), shape.dp * (CELL + 1));
    }
  }, [i.id, usesTmem, accDp, accCols, shapeId, fill]);

  return (
    <div class="panel">
      <h3>
        TMEM <small>— {TMEM_DP} DP × {TMEM_COLS} cols × uint32 (showing first {SHOWN_COLS})</small>
      </h3>
      <canvas ref={canvasRef} style={{ height: ROWS_PX }} />
      {usesTmem ? (
        <>
          <p class="panel__note">
            acc footprint: <code>{accDp} DP × {accCols} cols</code>
            {ctaGroup === 2 && ' (cta_group::2 halves M across 2 CTAs)'}
          </p>
          {layoutInfo ? (
            <p class="panel__note">
              PTX <code>{layoutInfo.name}</code>:{' '}
              <span class="tmem__conditions">{layoutInfo.conditions}</span>
              <span class="tmem__desc">{layoutInfo.description}</span>
            </p>
          ) : (
            <p class="panel__note panel__note--dim">
              (M={i.M}, cta={ctaGroup} not in PTX A–G taxonomy)
            </p>
          )}
          {range && (
            <p class="panel__note">
              viewing{' '}
              <code>{range.peerLabel}</code> · holds <code>M[{range.lo}..{range.hi})</code>
              {ctaGroup === 2 && (
                <>
                  {' '}
                  <button
                    class="tmem__peer-btn"
                    onClick={() => setPeer((p) => (p === 0 ? 1 : 0))}
                  >
                    switch peer
                  </button>
                </>
              )}
            </p>
          )}
          <label class="tmem__pick">
            tcgen05 shape
            <select value={shapeId} onChange={(e) => setShapeId((e.target as HTMLSelectElement).value)}>
              {TCGEN05_SHAPES.map((s) => (
                <option value={s.id}>{s.id}</option>
              ))}
            </select>
          </label>
          <div class="tmem__legend">
            <span class="tmem__swatch" style={{ background: subColor(0) }} /> subp 0
            <span class="tmem__swatch" style={{ background: subColor(1) }} /> subp 1
            <span class="tmem__swatch" style={{ background: subColor(2) }} /> subp 2
            <span class="tmem__swatch" style={{ background: subColor(3) }} /> subp 3
            <span class="tmem__swatch tmem__swatch--acc" /> acc
            <span class="tmem__swatch tmem__swatch--issue" /> ld/st issue
          </div>
        </>
      ) : (
        <p class="panel__note panel__note--dim">
          {i.family} does not use TMEM (accumulator lives in registers / <code>.reg</code>).
        </p>
      )}
    </div>
  );
}
