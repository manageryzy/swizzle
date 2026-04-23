import {
  activePattern,
  activeSwizzle,
  activeTileDims,
  currentAccesses,
  maxConflict,
} from '../state';
import { maxConflictWay } from '../patterns';
import { SWIZZLES, type SwizzleKind, effectiveSwizzle } from '../swizzle';
import { spec } from '../state';
import { TruthFooter } from './TruthFooter';

export function ConflictMeter() {
  const way = maxConflict.value;
  const sw = activeSwizzle.value;
  const pat = activePattern.value;
  const accesses = currentAccesses.value;
  const s = spec.value;

  const status = way === 1 ? 'none' : way <= 2 ? 'mild' : way <= 4 ? 'moderate' : 'severe';

  const elemBytes = activeTileDims.value.elemBytes;
  const allWays: { kind: SwizzleKind; way: number }[] = (
    Object.keys(SWIZZLES) as SwizzleKind[]
  ).map((k) => ({ kind: k, way: maxConflictWay(accesses, effectiveSwizzle(k, elemBytes)) }));
  const bestWay = Math.min(...allWays.map((x) => x.way));

  return (
    <div class="panel">
      <h3>
        Swizzle leaderboard <small>— tap a row to switch</small>
      </h3>
      <div class={`conflict-meter__way conflict-meter__way--${status}`}>
        {way === 1 ? 'no conflict · 1 cycle' : `${way}-way · ${way} cycles`}
      </div>
      <div class="conflict-meter__detail">
        pattern <code>{pat.id}</code> × <code>Swizzle&lt;{sw.B},{sw.M},{sw.S}&gt;</code>
      </div>

      <div class="conflict-meter__compare">
        <div class="conflict-meter__compare-head">
          all swizzles <span class="conflict-meter__compare-hint">· lower = better</span>
        </div>
        {allWays.map(({ kind, way: w }) => (
          <button
            key={kind}
            class={`conflict-meter__row ${kind === s.swizzle ? 'is-active' : ''} ${w === bestWay ? 'is-best' : ''}`}
            onClick={() => (spec.value = { ...s, swizzle: kind })}
            title={`switch to ${kind}`}
          >
            <span class="conflict-meter__row-kind">
              {kind}
              {w === bestWay && <span class="conflict-meter__best-tag">best</span>}
            </span>
            <span class="conflict-meter__row-bar">
              <span
                class="conflict-meter__row-fill"
                style={{ width: `${Math.min(100, (w / 32) * 100)}%` }}
              />
            </span>
            <span class="conflict-meter__row-way">
              {w === 1 ? '✓' : `${w}-way`}
            </span>
          </button>
        ))}
      </div>

      <TruthFooter
        verified
        models="per-swizzle max-way for the active pattern; element-size M upcast."
        schematic="leaderboard compares atoms at the same pattern — does not rank by actual cycle cost under replay."
        cite="swizzle.ts · effectiveSwizzle, patterns.ts · maxConflictWay"
      />
    </div>
  );
}
