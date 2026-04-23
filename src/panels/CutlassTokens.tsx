import { useState } from 'preact/hooks';
import {
  activeSwizzle,
  blkMMult,
  blkNMult,
  clusterShape,
  inst,
  kStages,
  spec,
  tileK,
} from '../state';
import { LayoutType, layoutTypeOf, type Swizzle } from '../swizzle';
import type { InstSpec } from '../instructions';
import { TruthFooter } from './TruthFooter';

// Translate the current UI config into the CUTLASS/cute tokens a user would
// write in their own code. Pure display — no side effects beyond the copy
// button.

function swizzleAtomName(major: 'K' | 'MN', sw: Swizzle): string {
  const suffix =
    sw.B === 0 ? 'INTER' :
    sw.B === 1 ? 'SW32' :
    sw.B === 2 && sw.M === 5 ? 'SW128_32B' :
    sw.B === 2 ? 'SW64' :
    sw.B === 3 ? 'SW128' : `Swizzle<${sw.B},${sw.M},${sw.S}>`;
  return `Layout_${major}_${suffix}_Atom`;
}

function archTag(archId: string): string {
  return `cutlass::arch::${archId[0].toUpperCase() + archId.slice(1)}`;
}

function kernelSchedule(i: InstSpec): string {
  if (i.family === 'wgmma') return 'KernelTmaWarpSpecialized{Pingpong|Cooperative}';
  if (i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled')
    return `KernelTmaWarpSpecialized${i.warpSpecialized ? 'WarpSpecialized' : 'Cooperative'}Sm100`;
  if (i.family === 'mma') return 'KernelCpAsyncWarpSpecialized (cp.async staging)';
  if (i.family === 'wmma') return 'KernelMultistage (sm_70+ legacy path)';
  return '—';
}

export function CutlassTokens() {
  const i = inst.value;
  const s = spec.value;
  const sw = activeSwizzle.value;
  const [copied, setCopied] = useState(false);

  const descArch = i.arch === 'sm100' ? 'sm100' : 'sm90';
  const ltEnum = (() => {
    try {
      const lt = layoutTypeOf(sw);
      return `${descArch === 'sm100' ? 'SmemDescriptor' : 'GmmaDescriptor'}::LayoutType::${LayoutType[lt]}`;
    } catch {
      return '—';
    }
  })();

  const copyLink = async () => {
    await navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div class="panel cutlass">
      <h3>
        CUTLASS tokens
        <button class="cutlass__copy" onClick={copyLink}>
          {copied ? '✓ copied' : 'copy link'}
        </button>
      </h3>
      <dl class="cutlass__grid">
        <dt>arch tag</dt>
        <dd><code>{archTag(i.arch)}</code></dd>

        <dt>TileShape</dt>
        <dd>
          <code>
            Shape&lt;_{i.M * blkMMult.value}, _{i.N * blkNMult.value}, _{tileK.value}&gt;
          </code>
        </dd>

        <dt>ClusterShape</dt>
        <dd>
          <code>
            Shape&lt;_{clusterShape.value[0]}, _{clusterShape.value[1]}, _{clusterShape.value[2]}&gt;
          </code>
        </dd>

        <dt>StageCount</dt>
        <dd>
          <code>cutlass::gemm::collective::StageCountAuto /* = {kStages.value} */</code>
        </dd>

        <dt>inst atom</dt>
        <dd><code>{i.mnemonic}</code></dd>

        <dt>
          swizzle atom (A)
          <span class="truth__verified">verified vs cute</span>
        </dt>
        <dd><code>{swizzleAtomName(s.majorA, sw)}&lt;ElementA&gt;</code></dd>

        <dt>
          swizzle atom (B)
          <span class="truth__verified">verified vs cute</span>
        </dt>
        <dd><code>{swizzleAtomName(s.majorB, sw)}&lt;ElementB&gt;</code></dd>

        <dt>
          Swizzle&lt;B,M,S&gt;
          <span class="truth__verified">verified vs cute</span>
        </dt>
        <dd><code>Swizzle&lt;{sw.B},{sw.M},{sw.S}&gt;</code></dd>

        <dt>descriptor layout_type</dt>
        <dd><code>{ltEnum}</code></dd>

        <dt>A operand source</dt>
        <dd>
          <code>
            {s.aSource === 'smem' ? 'SS (smem desc)' :
             s.aSource === 'rmem' ? 'RS (register fragment)' :
             'TS (tmem desc)'}
          </code>
        </dd>

        <dt>kernel schedule</dt>
        <dd><code>{kernelSchedule(i)}</code></dd>

        <dt>collective builder</dt>
        <dd>
          <code>
            CollectiveBuilder&lt;{archTag(i.arch).split('::').pop()}, OpClassTensorOp,
            ElementA, Layout{s.majorA}, …, ClusterShape_{i.ctaGroup ?? 1}x1x1, …&gt;
          </code>
        </dd>
      </dl>
      <p class="cutlass__foot">
        builder lives under <code>cutlass/gemm/collective/builders/</code>; each
        arch has its own dispatch file (sm90_*, sm100_*). Cite:{' '}
        <code>{i.source}</code>. The CTA tile is carved out of a GMEM tensor
        via <code>local_tile(mA, make_shape(BLK_M, BLK_K), tile_coord)</code>{' '}
        (<code>cute/algorithm/tile.hpp</code>) and fanned across atoms via{' '}
        <code>TiledMMA&lt;AtomLayoutMNK&gt;</code> (<code>cute/atom/atom.hpp</code>).
      </p>
      <TruthFooter
        verified
        models="arch tag, inst mnemonic, cute swizzle atom name, Swizzle<B,M,S> tuple (verified vs cute), ClusterShape, A operand source (SS/RS/TS), ctaGroup"
        schematic="kernel schedule / collective builder signature — guidance, not a compile-checked template"
      />
    </div>
  );
}
