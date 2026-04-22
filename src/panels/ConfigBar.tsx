import { activePatternId, commitConfig, commitMode, inst, modeConfig, spec, wgmmaShowAllShapes } from '../state';
import { SWIZZLES, type SwizzleKind } from '../swizzle';
import { PATTERNS } from '../patterns';
import {
  ctaGroupsForMode,
  dtypesForMode,
  modeOfInst,
  validShapesFor,
  type Mode,
} from '../inst_resolver';
import type { Dtype, Major, OperandSource } from '../instructions';

const MODES: Mode[] = ['tcgen05', 'wgmma', 'mma', 'wmma'];
const MODE_LABEL: Record<Mode, string> = {
  tcgen05: 'tcgen05 · sm_100',
  wgmma: 'wgmma · sm_90',
  mma: 'mma · sm_80+',
  wmma: 'wmma · sm_70+',
};

export function ConfigBar() {
  const i = inst.value;
  const s = spec.value;
  const cfg = modeConfig.value;
  const mode = modeOfInst(i);
  const dtypes = dtypesForMode(mode);
  const ctas = ctaGroupsForMode(mode);

  const showAllWgmma = wgmmaShowAllShapes.value;
  // Hide PTX-only wgmma shapes unless the user opted in. Other modes are
  // unaffected since they only ship `cutlass-atom` entries.
  const includePtxOnly = mode !== 'wgmma' || showAllWgmma;
  const validShapes = validShapesFor(
    mode,
    cfg.dtypeA,
    cfg.dtypeB,
    cfg.accDtype,
    cfg.ctaGroup ?? 1,
    !!cfg.sparse,
    !!cfg.warpSpecialized,
    includePtxOnly,
  );
  const shapeSourceForN = new Map(
    validShapes.filter((x) => x.M === cfg.M).map((x) => [x.N, x.shapeSource]),
  );
  const Ms = uniq(validShapes.map((x) => x.M));
  const Ns = uniq(validShapes.filter((x) => x.M === cfg.M).map((x) => x.N));
  const Ks = uniq(validShapes.filter((x) => x.M === cfg.M && x.N === cfg.N).map((x) => x.K));

  return (
    <div class="configbar-v2">
      {/* Mode tabs */}
      <div class="configbar-v2__tabs">
        {MODES.map((m) => (
          <button
            class={`configbar-v2__tab ${mode === m ? 'is-active' : ''}`}
            onClick={() => commitMode(m)}
            title={MODE_LABEL[m]}
          >
            {m}
          </button>
        ))}
      </div>

      <div class="configbar-v2__rows">
        <div class="configbar-v2__group">
          <span class="configbar-v2__label">shape</span>
          <Select
            label="M"
            value={cfg.M}
            options={Ms}
            onChange={(v) => commitConfig({ M: v })}
          />
          <Select
            label="N"
            value={cfg.N}
            options={Ns}
            display={(n) => {
              const src = shapeSourceForN.get(n);
              return src === 'ptx-only' ? `${n} [PTX-only]` : String(n);
            }}
            onChange={(v) => commitConfig({ N: v })}
          />
          <Select
            label="K"
            value={cfg.K}
            options={Ks}
            onChange={(v) => commitConfig({ K: v })}
          />
        </div>

        <div class="configbar-v2__group">
          <span class="configbar-v2__label">dtype</span>
          <Select
            label="A"
            value={cfg.dtypeA}
            options={dtypes.a}
            onChange={(v) => commitConfig({ dtypeA: v as Dtype })}
          />
          <span class="configbar-v2__cross">×</span>
          <Select
            label="B"
            value={cfg.dtypeB}
            options={dtypes.b}
            onChange={(v) => commitConfig({ dtypeB: v as Dtype })}
          />
          <span class="configbar-v2__cross">→</span>
          <Select
            label="acc"
            value={cfg.accDtype}
            options={dtypes.acc}
            onChange={(v) => commitConfig({ accDtype: v as Dtype })}
          />
        </div>

        <div class="configbar-v2__group">
          <span class="configbar-v2__label">layout</span>
          <Select
            label="A"
            value={cfg.majorA}
            options={i.majorA as Major[]}
            display={(m) => `${m}-major${m === 'K' ? ' (T)' : ' (N)'}`}
            onChange={(v) => commitConfig({ majorA: v as Major })}
          />
          <Select
            label="B"
            value={cfg.majorB}
            options={i.majorB as Major[]}
            display={(m) => `${m}-major${m === 'K' ? ' (T)' : ' (N)'}`}
            onChange={(v) => commitConfig({ majorB: v as Major })}
          />
        </div>

        {i.aSource.length > 1 && (
          <div class="configbar-v2__group">
            <span class="configbar-v2__label">A source</span>
            <Select
              label=""
              value={cfg.aSource ?? i.aSource[0]}
              options={i.aSource as OperandSource[]}
              display={(src) => {
                // wgmma: SS vs RS — SMEM-sourced A vs register-sourced A.
                // tcgen05: SS vs TS — SMEM vs TMEM.
                if (src === 'smem') return 'SMEM (SS)';
                if (src === 'rmem') return 'RMEM (RS)';
                if (src === 'tmem') return 'TMEM (TS)';
                return src;
              }}
              onChange={(v) => commitConfig({ aSource: v as OperandSource })}
            />
          </div>
        )}

        <div class="configbar-v2__group">
          <span class="configbar-v2__label">swizzle</span>
          <Select
            label=""
            value={s.swizzle}
            options={Object.keys(SWIZZLES) as SwizzleKind[]}
            onChange={(v) => (spec.value = { ...s, swizzle: v as SwizzleKind })}
          />
        </div>

        <div class="configbar-v2__group">
          <span class="configbar-v2__label">pattern</span>
          <Select
            label=""
            value={activePatternId.value}
            options={Object.keys(PATTERNS)}
            display={(id) => PATTERNS[id as string].name}
            onChange={(v) => (activePatternId.value = v as string)}
          />
        </div>

        {mode === 'wgmma' && (
          <div class="configbar-v2__group">
            <span class="configbar-v2__label">shapes</span>
            <Check
              label="show all PTX shapes"
              checked={showAllWgmma}
              onChange={(v) => (wgmmaShowAllShapes.value = v)}
            />
          </div>
        )}

        {mode === 'tcgen05' && (
          <div class="configbar-v2__group">
            <span class="configbar-v2__label">flags</span>
            {ctas.length > 1 && (
              <Select
                label="cta_group"
                value={cfg.ctaGroup ?? 1}
                options={ctas}
                display={(n) => `cta_group::${n}`}
                onChange={(v) => commitConfig({ ctaGroup: Number(v) as 1 | 2 })}
              />
            )}
            <Check
              label="sparse .sp"
              checked={!!cfg.sparse}
              onChange={(v) => commitConfig({ sparse: v })}
            />
            <Check
              label="ws .ws"
              checked={!!cfg.warpSpecialized}
              onChange={(v) => commitConfig({ warpSpecialized: v })}
            />
          </div>
        )}
      </div>

      <div class="configbar-v2__resolved">
        resolves to <code>{i.id}</code> · <code>{i.mnemonic}</code>
      </div>
    </div>
  );
}

function uniq<T>(xs: T[]): T[] {
  return [...new Set(xs)];
}

function Select<T extends string | number>({
  label,
  value,
  options,
  display,
  onChange,
}: {
  label: string;
  value: T;
  options: T[];
  display?: (v: T) => string;
  onChange: (v: T) => void;
}) {
  return (
    <label class="configbar-v2__field">
      {label && <span class="configbar-v2__field-label">{label}</span>}
      <select
        value={String(value)}
        onChange={(e) => {
          const raw = (e.target as HTMLSelectElement).value;
          const parsed = (typeof value === 'number' ? Number(raw) : raw) as T;
          onChange(parsed);
        }}
      >
        {options.map((o) => (
          <option value={String(o)}>{display ? display(o) : String(o)}</option>
        ))}
      </select>
    </label>
  );
}

function Check({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label class="configbar-v2__check">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange((e.target as HTMLInputElement).checked)}
      />
      <span>{label}</span>
    </label>
  );
}
