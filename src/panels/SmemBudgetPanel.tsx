import { currentKStep, currentPhase, inst, kStages, phaseProgress, tileK } from '../state';
import { buildBudget, kIterations, maxStages, type Segment } from '../smem_budget';

const COLOR: Record<Segment['kind'], string> = {
  A: '#7ec699',
  B: '#7e9cd8',
  pad: '#3a3a3a',
  mbar: '#b48ead',
  unused: '#1a1f2a',
};

export function SmemBudgetPanel() {
  const i = inst.value;
  const phase = currentPhase.value;
  const progress = phaseProgress.value;
  const kStep = currentKStep.value;
  const stages = kStages.value;
  const tileKValue = tileK.value;
  const budget = buildBudget(i, stages);
  const iters = kIterations(i, tileKValue);
  const maxS = maxStages(i);

  // During mma.step k=X, consumer reads stage = (X % stages); producer is
  // loading stage ((X + stages - 1) % stages). Both respect CUTLASS rotation.
  const activeStageConsumer = (() => {
    if (!phase || !kStep) return null;
    if (phase.kind === 'wgmma.step' || phase.kind === 'tcgen05.mma.step') {
      return kStep.k % stages;
    }
    if (phase.kind === 'tma.load' || phase.kind === 'cp.async') {
      // During load we highlight the stage being filled.
      return Math.floor(progress * stages) % stages;
    }
    return null;
  })();

  return (
    <div class="panel smem-budget">
      <h3>
        SMEM budget{' '}
        <small>
          — {(budget.total / 1024).toFixed(0)} KiB/SM · <code>kStages={stages}</code> ·{' '}
          {(budget.usedBytes / 1024).toFixed(1)} KiB used{' '}
          {budget.fits ? (
            <span class="budget__fits">fits</span>
          ) : (
            <span class="budget__overflow">OVERFLOW — max {maxS} stages</span>
          )}
        </small>
      </h3>

      <div class="budget__bar">
        {budget.segments.map((seg, idx) => {
          const pct = (seg.bytes / budget.total) * 100;
          const isActive =
            activeStageConsumer !== null &&
            (seg.kind === 'A' || seg.kind === 'B') &&
            seg.stage === activeStageConsumer;
          return (
            <div
              key={`${seg.kind}.${seg.stage ?? 0}.${idx}`}
              class={`budget__seg budget__seg--${seg.kind} ${isActive ? 'is-active' : ''}`}
              style={{ width: `${pct}%`, background: COLOR[seg.kind] }}
              title={`${seg.label} · ${seg.bytes} B (${pct.toFixed(2)}%)`}
            >
              <span class="budget__seg-label">{seg.label}</span>
            </div>
          );
        })}
      </div>

      <div class="budget__controls">
        <label>
          kStages
          <select value={stages} onChange={(e) => (kStages.value = Number((e.target as HTMLSelectElement).value))}>
            {[1, 2, 3, 4, 5, 6, 7, 8].map((s) => (
              <option value={s} disabled={s > maxS}>
                {s}
                {s === maxS ? ' (max)' : ''}
              </option>
            ))}
          </select>
        </label>
        <label>
          tileK
          <select value={tileKValue} onChange={(e) => (tileK.value = Number((e.target as HTMLSelectElement).value))}>
            {[1, 2, 4, 8, 16, 32].map((mult) => (
              <option value={i.K * mult}>
                {i.K * mult} ({mult} iter)
              </option>
            ))}
          </select>
        </label>
        <span class="budget__foot">
          stage: A = {budget.stageBytes.a} B, B = {budget.stageBytes.b} B, pad ={' '}
          {budget.stageBytes.pad} B, mbar = {budget.stageBytes.mbar} B → stage_bytes ={' '}
          {budget.stageBytes.stage} B · K iter = {iters} · consumer stage:{' '}
          <code>{activeStageConsumer ?? '—'}</code>
        </span>
      </div>
      <p class="budget__legend">
        <span class="budget__swatch" style={{ background: COLOR.A }} /> A tile
        <span class="budget__swatch" style={{ background: COLOR.B }} /> B tile
        <span class="budget__swatch" style={{ background: COLOR.pad }} /> 128 B pad (A+B aligned as a pair)
        <span class="budget__swatch" style={{ background: COLOR.mbar }} /> mbar (PipelineTmaAsync state)
      </p>
    </div>
  );
}
