import { currentKStep, inst, kStages, spec, stageAxis } from '../state';
import { SmemTileView } from './SmemTileView';
import { canonicalLayout, canonicalT } from '../gmma_layouts';
import { activeSwizzle } from '../state';
import { bytesOf } from '../smem_budget';
import { ringState } from '../pipeline_state';

type AnySource = 'smem' | 'rmem' | 'tmem' | 'gmem-wmma';

function sourceOfOperand(
  aSource: readonly AnySource[] | AnySource[],
  operand: 'A' | 'B',
  userSource: string,
  bSource: readonly AnySource[] | AnySource[],
): AnySource {
  if (operand === 'A') {
    if ((aSource as AnySource[]).includes(userSource as AnySource))
      return userSource as AnySource;
    return (aSource as AnySource[])[0];
  }
  const b = bSource as AnySource[];
  return b.includes('smem') ? 'smem' : b[0];
}

function OperandRow({ operand }: { operand: 'A' | 'B' }) {
  const i = inst.value;
  const s = spec.value;
  const stages = kStages.value;
  const axis = stageAxis.value;
  const kStep = currentKStep.value;
  const src = sourceOfOperand(i.aSource, operand, s.aSource, i.bSource);
  const inSmem = src === 'smem';

  const major = operand === 'A' ? s.majorA : s.majorB;
  const dtype = operand === 'A' ? i.aDtypes[0] : i.bDtypes[0];
  const elemBytes = bytesOf(dtype);

  const descArch: 'sm90' | 'sm100' = i.arch === 'sm100' ? 'sm100' : 'sm90';
  const needsDesc =
    i.family === 'wgmma' || i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled';
  const canon = needsDesc && inSmem ? canonicalLayout(descArch, major, s.swizzle) : null;

  // Real CUTLASS ring state: stage s holds K slice
  //   head - ((head - s) mod kStages),  head = iter + kStages - 1
  // Producer warpgroup is ahead of consumer; at steady state, one stage is
  // being consumed, one is being filled, others hold different K slices.
  const iter = kStep?.k ?? 0;
  const ring = ringState(iter, stages);

  return (
    <div class="smem-operand">
      <div class="smem-operand__head">
        <span class="smem-operand__name">operand {operand}</span>
        <span class="smem-operand__meta">
          {operand === 'A' ? i.M : i.N} × {i.K} · {dtype}
          {inSmem ? (
            <>
              {' '}· <code class="pill pill--smem">SMEM</code>
              {' '}· <code>kStages = {stages}</code>
              {' '}· consumer iter <code>{iter}</code>
            </>
          ) : (
            <>
              {' '}· <code class={`pill pill--${src}`}>{src.toUpperCase()}</code> (not in SMEM)
            </>
          )}
        </span>
      </div>
      {inSmem ? (
        <>
          <div class={`stage-grid stage-grid--${axis}`}>
            {ring.map((st) => (
              <div
                key={`stage-${operand}-${st.stage}`}
                class={`stage ${st.role === 'consume' ? 'is-consumer' : ''} ${st.role === 'fill' ? 'is-producer' : ''}`}
              >
                <div class="stage__badge">
                  <span>stage {st.stage} · slice {st.slice}</span>
                  {st.role !== 'hold' && <span class="stage__tag">{st.role}</span>}
                </div>
                <div class="smem-operand__views">
                  <SmemTileView operand={operand} view="logical" sliceIdx={st.slice} />
                  <SmemTileView operand={operand} view="physical" sliceIdx={st.slice} />
                </div>
              </div>
            ))}
          </div>
          {canon && (
            <div class="smem-operand__canon">
              <code>
                {canon.swizzle} ∘ {canon.shape} : {canon.stride}
              </code>
              <span> · T = {canonicalT(elemBytes)}</span>
            </div>
          )}
        </>
      ) : (
        <div class="smem-operand__empty">
          operand {operand} lives in <code>{src}</code> — no SMEM footprint for this instruction
        </div>
      )}
    </div>
  );
}

export function SmemPanel() {
  const sw = activeSwizzle.value;
  const stages = kStages.value;
  const axis = stageAxis.value;
  return (
    <div class="panel panel--wide smem-panel">
      <h3>
        SMEM tiles{' '}
        <small>
          logical (pre-swizzle) vs physical (post-swizzle) ·{' '}
          <code>Swizzle&lt;{sw.B},{sw.M},{sw.S}&gt;</code> · {stages} stages
        </small>
        <label class="smem-panel__axis">
          stack
          <select
            value={axis}
            onChange={(e) => (stageAxis.value = (e.target as HTMLSelectElement).value as any)}
          >
            <option value="vertical">vertical (along M)</option>
            <option value="horizontal">horizontal (along K)</option>
          </select>
        </label>
      </h3>
      <OperandRow operand="A" />
      <OperandRow operand="B" />
      <div class="legend">
        <span class="legend__item">
          stage <span class="legend__swatch legend__swatch--consume" /> consumed by mma
        </span>
        <span class="legend__item">
          <span class="legend__swatch legend__swatch--fill" /> filled by TMA / cp.async
        </span>
        <span class="legend__item">ring size = kStages; same cell colour traces a matrix element</span>
      </div>
    </div>
  );
}
