import { useEffect, useRef, useMemo } from 'preact/hooks';
import { currentKStep, currentPhase, inst, phaseProgress, spec } from '../state';
import { clayoutOf, ownershipMap } from '../cute_mma_layouts';
import { layoutAt } from '../cute_layout';

// Register-file / accumulator panel. Renders the output tile (M × N) with
// cells coloured by the thread that owns them in the warp's .reg state space.
// (Cute calls this "rmem"; PTX itself has no RMEM — the storage is `.reg`
// plus spills to `.local`.) Approximation where no CLayout is ported — see
// mma_traits_sm{80,90_gmma,100}.hpp for exact.

function threadColor(t: number, totalThreads: number): string {
  return `hsl(${(t * 360) / totalThreads}, 75%, 55%)`;
}

// Fallback owner map for instructions without a ported CLayout (wmma, tcgen05).
// Plausible shape only — the real fragment comes from mma_traits_sm{70,100}.hpp.
function ownerFallback(m: number, n: number, threadsPerMma: number): number {
  const quad = (m % 8) >> 1;
  const warp = (m >> 3) & 3;
  const col = n & 7;
  return (warp * 32 + quad * 8 + col) % threadsPerMma;
}

export function RmemPanel() {
  const i = inst.value;
  const s = spec.value;
  const phase = currentPhase.value;
  const progress = phaseProgress.value;
  const kStep = currentKStep.value;
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const isWarpGroup = i.family === 'wgmma' || i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled';
  const threadsPerMma = i.family === 'wmma' || i.family === 'mma' ? 32 : 128;
  const clayout = clayoutOf(i.id);
  const realOwner = clayout ? ownershipMap(clayout) : null;
  const ownerOf = (m: number, n: number) =>
    realOwner ? realOwner(m, n) : ownerFallback(m, n, threadsPerMma);

  // Thread-first fill order. Lane 0 writes its 4 cells, then lane 1 writes its
  // 4 cells, etc. This produces a distinctly "warp-issuing-writes" animation
  // instead of a row-major sweep.
  const fillOrder = useMemo(() => {
    const out: Array<{ m: number; n: number }> = [];
    if (clayout) {
      for (let tid = 0; tid < clayout.threads; tid++) {
        for (let vid = 0; vid < clayout.valuesPerThread; vid++) {
          const idx = tid + vid * clayout.threads;
          const off = layoutAt(clayout.shape, clayout.stride, idx);
          const m = off % clayout.M;
          const n = Math.floor(off / clayout.M);
          out.push({ m, n });
        }
      }
    } else {
      // Fallback: group by approximate owner then step through values.
      for (let t = 0; t < threadsPerMma; t++) {
        for (let m = 0; m < i.M; m++)
          for (let n = 0; n < i.N; n++)
            if (ownerFallback(m, n, threadsPerMma) === t) out.push({ m, n });
      }
    }
    return out;
  }, [clayout?.label, threadsPerMma, i.M, i.N, i.id]);

  // Accumulator fill level ∈ [0, 1]. During mma.step k=X, climb from k/total
  // to (k+1)/total; stays full after the last step.
  const fill = (() => {
    if (phase?.kind === 'wgmma.step' || phase?.kind === 'tcgen05.mma.step') {
      if (!kStep) return 0;
      return Math.min(1, (kStep.k + progress) / kStep.total);
    }
    const isAfterMma = phase && (phase.kind === 'tcgen05.ld' || phase.kind === 'epilogue.tma.store');
    return isAfterMma ? 1 : 0;
  })();

  // Clamp display to something drawable.
  const rows = Math.min(i.M, 64);
  const cols = Math.min(i.N, 64);
  const cell = rows > 32 ? 6 : 10;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cols * cell;
    const H = rows * cell;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // First fill the whole tile dark.
    ctx.fillStyle = '#0d1017';
    ctx.fillRect(0, 0, cols * cell, rows * cell);

    // Fill in (tid, vid) order up to the current progress.
    const filledCount = Math.floor(fill * fillOrder.length);
    for (let idx = 0; idx < filledCount && idx < fillOrder.length; idx++) {
      const { m, n } = fillOrder[idx];
      if (m >= rows || n >= cols) continue;
      ctx.fillStyle = threadColor(ownerOf(m, n), threadsPerMma);
      ctx.fillRect(n * cell, m * cell, cell - 1, cell - 1);
    }

    // Bright "pen tip" at the currently-writing cell.
    if (fill > 0 && fill < 1 && filledCount < fillOrder.length) {
      const { m, n } = fillOrder[filledCount] ?? { m: 0, n: 0 };
      if (m < rows && n < cols) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(n * cell, m * cell, cell - 1, cell - 1);
      }
    }
  }, [rows, cols, cell, threadsPerMma, i.id, fill, fillOrder]);

  return (
    <div class="panel">
      <h3>
        {i.accIn === 'tmem' ? 'TMEM acc' : 'Registers (.reg)'}
        <small>
          {' '}
          — {i.M}×{i.N} tile · {threadsPerMma} threads · colour = owner
          {i.accIn === 'rmem' && ' · cute calls this "rmem"'}
        </small>
      </h3>
      <canvas ref={canvasRef} />
      <p class="panel__note">
        {isWarpGroup ? 'warpgroup (4 warps × 32 lanes)' : 'single warp (32 lanes)'} ·{' '}
        A source: <code class={`pill--src ${s.aSource === 'tmem' ? 'pill--tmem' : ''}`}>{s.aSource.toUpperCase()}</code>
        {i.family === 'wgmma' && (
          <>
            {' '}· variant:{' '}
            <code>{s.aSource === 'rmem' ? 'RS (reg fragment + smem)' : 'SS (smem desc + smem desc)'}</code>
          </>
        )}
      </p>
      <p class={`panel__note ${realOwner ? '' : 'panel__note--dim'}`}>
        fragment layout:{' '}
        {clayout ? (
          <>
            <code>{clayout.label}</code> (exact — from <code>{clayout.sourceFile}</code>)
          </>
        ) : i.accIn === 'tmem' ? (
          <>
            tcgen05 accumulator lives in <code>TMEM</code> — see the TMEM panel
            for the per-DP / per-col mapping (PTX Layouts A–G). This owner view
            is approximate since sm_100 doesn't distribute C across threads.
          </>
        ) : (
          <>approximate (no ported CLayout for {i.id})</>
        )}
      </p>
      {rows < i.M || cols < i.N ? (
        <p class="panel__note panel__note--dim">
          (display clipped to 64×64; inst is {i.M}×{i.N})
        </p>
      ) : null}
    </div>
  );
}
