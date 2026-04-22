import { activePattern, activeSwizzle, activeTileDims, currentAccesses, currentConflicts, maxConflict } from '../state';
import { maxConflictWay } from '../patterns';
import { SWIZZLES, type SwizzleKind, effectiveSwizzle } from '../swizzle';
import { spec } from '../state';

export function ConflictMeter() {
  const cs = currentConflicts.value;
  const way = maxConflict.value;
  const sw = activeSwizzle.value;
  const pat = activePattern.value;
  const accesses = currentAccesses.value;
  const s = spec.value;

  const status = way === 1 ? 'none' : way <= 2 ? 'mild' : way <= 4 ? 'moderate' : 'severe';

  // Per-swizzle leaderboard for the current pattern. Use the element-size of
  // the active operand so the ranking matches what the SMEM panel renders
  // (fp8/fp32 shift M and change the conflict picture).
  const elemBytes = activeTileDims.value.elemBytes;
  const allWays: { kind: SwizzleKind; way: number }[] = (
    Object.keys(SWIZZLES) as SwizzleKind[]
  ).map((k) => ({ kind: k, way: maxConflictWay(accesses, effectiveSwizzle(k, elemBytes)) }));
  const bestWay = Math.min(...allWays.map((x) => x.way));

  return (
    <div class="panel">
      <h3>
        Conflicts <small>— max lanes colliding on a single bank</small>
      </h3>

      <div class="conflict-meter">
        <div class={`conflict-meter__way conflict-meter__way--${status}`}>
          {way === 1 ? 'no conflict' : `${way}-way`}
        </div>
        <div class="conflict-meter__detail">
          pattern <code>{pat.id}</code> × <code>Swizzle&lt;{sw.B},{sw.M},{sw.S}&gt;</code>
        </div>
      </div>

      <p class="conflict-meter__desc">
        pattern <code>{pat.name}</code>: {pat.description}
      </p>

      <div class="conflict-meter__compare">
        <div class="conflict-meter__compare-head">all swizzles</div>
        {allWays.map(({ kind, way: w }) => (
          <button
            key={kind}
            class={`conflict-meter__row ${kind === s.swizzle ? 'is-active' : ''} ${w === bestWay ? 'is-best' : ''}`}
            onClick={() => (spec.value = { ...s, swizzle: kind })}
            title={`switch to ${kind}`}
          >
            <span class="conflict-meter__row-kind">{kind}</span>
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

      {cs.length > 0 && (
        <div class="conflict-meter__banks">
          <div class="conflict-meter__banks-head">top hot banks</div>
          <ul>
            {cs.slice(0, 5).map((c) => (
              <li>
                bank <code>{c.bank}</code> ← {c.way} lanes: [{c.lanes.slice(0, 6).join(', ')}
                {c.lanes.length > 6 ? '…' : ''}]
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
