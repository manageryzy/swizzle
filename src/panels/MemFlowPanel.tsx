// Memory-hierarchy flow diagram. Shows the end-to-end data path a CUTLASS
// kernel takes each K iteration, with the active phase highlighted:
//
//   GMEM(A) ──tma.load──> SMEM(A) ──ldmatrix / desc──> reg(A) ─┐
//                                                              ├── MMA ── acc (.reg / TMEM)
//   GMEM(B) ──tma.load──> SMEM(B) ──ldmatrix / desc──> reg(B) ─┘         │
//                                                                        ▼
//                                     SMEM(C) <── stmatrix / tcgen05.st / stg ──
//                                       │
//                                       └── tma.store ──> GMEM(C)
//
// Boxes glow when the current phase uses that state; arrows glow when the
// current phase is the operation on that edge. Scrubbing the Timeline
// animates the glow to tell the memory-movement story.

import {
  currentConsumerPhase,
  currentEpiloguePhase,
  currentProducerPhase,
  inst,
  pipelineMode,
  producerPhaseProgress,
  epiloguePhaseProgress,
  summary,
  world,
} from '../state';
import { TruthFooter } from './TruthFooter';

type NodeId =
  | 'gmemA' | 'gmemB' | 'gmemC'
  | 'smemA' | 'smemB' | 'smemC'
  | 'regA' | 'regB'
  | 'acc'
  | 'mma'
  // Phase 6 variant-specific nodes
  | 'tmemA'                // TS variant: A lives in TMEM
  | 'gmemMeta' | 'smemMeta'      // sparse: 2:4 metadata
  | 'gmemSFA' | 'smemSFA'         // block_scaled: SFA
  | 'gmemSFB' | 'smemSFB';        // block_scaled: SFB

type EdgeId =
  | 'tmaA' | 'tmaB'        // GMEM→SMEM
  | 'ldA' | 'ldB'          // SMEM→reg
  | 'descA' | 'descB'      // SMEM→MMA (via descriptor, no reg copy)
  | 'mmaOut'               // MMA writes acc
  | 'tmemLd'               // TMEM→reg (tcgen05.ld)
  | 'stgSmem'              // reg→SMEM (stmatrix)
  | 'tmaStore'             // SMEM→GMEM
  // Phase 6 variant-specific edges
  | 'tcgen05cp'            // TS variant: SMEM-A → TMEM-A (tcgen05.cp)
  | 'metaLoad' | 'metaDesc'         // sparse: GMEM→SMEM→MMA metadata
  | 'sfaLoad' | 'sfaDesc'            // block_scaled: SFA
  | 'sfbLoad' | 'sfbDesc';

interface Node { id: NodeId; x: number; y: number; w: number; h: number; label: string; sub?: string; cls: string; }
interface Edge { id: EdgeId; from: NodeId; to: NodeId; label: string; }

const NODE_W = 88;
const NODE_H = 44;

// Small-rect size for auxiliary nodes (metadata / scale tensors).
const AUX_W = 68;
const AUX_H = 22;

const NODES: Node[] = [
  // col 0: GMEM
  { id: 'gmemA', x: 20,  y: 20,  w: NODE_W, h: NODE_H, label: 'GMEM', sub: 'A', cls: 'mf-node--gmem' },
  { id: 'gmemB', x: 20,  y: 100, w: NODE_W, h: NODE_H, label: 'GMEM', sub: 'B', cls: 'mf-node--gmem' },
  { id: 'gmemC', x: 20,  y: 220, w: NODE_W, h: NODE_H, label: 'GMEM', sub: 'C', cls: 'mf-node--gmem' },
  // col 1: SMEM (ring)
  { id: 'smemA', x: 170, y: 20,  w: NODE_W, h: NODE_H, label: 'SMEM', sub: 'A · kStages', cls: 'mf-node--smem' },
  { id: 'smemB', x: 170, y: 100, w: NODE_W, h: NODE_H, label: 'SMEM', sub: 'B · kStages', cls: 'mf-node--smem' },
  { id: 'smemC', x: 170, y: 220, w: NODE_W, h: NODE_H, label: 'SMEM', sub: 'C epi', cls: 'mf-node--smem' },
  // col 2: Registers / MMA input
  { id: 'regA',  x: 320, y: 20,  w: NODE_W, h: NODE_H, label: '.reg', sub: 'A frag', cls: 'mf-node--reg' },
  { id: 'regB',  x: 320, y: 100, w: NODE_W, h: NODE_H, label: '.reg', sub: 'B frag', cls: 'mf-node--reg' },
  // col 3: MMA + accumulator
  { id: 'mma',   x: 470, y: 60,  w: NODE_W, h: NODE_H, label: 'MMA', sub: 'atom', cls: 'mf-node--mma' },
  { id: 'acc',   x: 620, y: 60,  w: NODE_W, h: NODE_H, label: 'acc', sub: '.reg / TMEM', cls: 'mf-node--acc' },
  // Phase 6 — TS variant: TMEM A region sits between SMEM-A and MMA. Drawn
  // in the .reg column so the tmemA→mma edge travels left-to-right.
  { id: 'tmemA', x: 320, y: 170, w: NODE_W, h: NODE_H, label: 'TMEM', sub: 'A region', cls: 'mf-node--tmem' },
  // Phase 6 — sparse: metadata tensor, tucked below B row for vertical grouping.
  { id: 'gmemMeta', x: 34, y: 172, w: AUX_W, h: AUX_H, label: 'GMEM meta', cls: 'mf-node--gmem mf-node--aux' },
  { id: 'smemMeta', x: 180, y: 172, w: AUX_W, h: AUX_H, label: 'SMEM meta', cls: 'mf-node--smem mf-node--aux' },
  // Phase 6 — block_scaled: SFA + SFB. SFA near A row, SFB near B row.
  { id: 'gmemSFA', x: 34, y: 68, w: AUX_W, h: AUX_H, label: 'GMEM SFA', cls: 'mf-node--gmem mf-node--aux' },
  { id: 'smemSFA', x: 180, y: 68, w: AUX_W, h: AUX_H, label: 'SMEM SFA', cls: 'mf-node--smem mf-node--aux' },
  { id: 'gmemSFB', x: 34, y: 148, w: AUX_W, h: AUX_H, label: 'GMEM SFB', cls: 'mf-node--gmem mf-node--aux' },
  { id: 'smemSFB', x: 180, y: 148, w: AUX_W, h: AUX_H, label: 'SMEM SFB', cls: 'mf-node--smem mf-node--aux' },
];

// The labels on load/store edges depend on which instruction family is
// active. sm_90 wgmma / sm_100 tcgen05 use cp.async.bulk.tensor (TMA);
// sm_80 mma uses plain cp.async; sm_70 wmma has no async copy at all
// (ld.shared / ld.global into the fragment directly). The shape of the
// epilogue also varies: TMA store for Hopper/Blackwell, stg / st.global
// for Ampere and older.
//
// `nodesForFamily` and `edgesForFamily` return just the data paths that are
// real for the active instruction. Nodes/edges that don't apply are omitted
// rather than drawn greyed-out, because "greyed" would still imply they
// participate in the data flow for that ISA.

type FamilyShape = 'tma-warpspec' | 'cpasync-mma' | 'wmma-direct';

function familyShape(family: string): FamilyShape {
  if (family === 'wgmma' || family === 'tcgen05' || family === 'tcgen05.block_scaled') return 'tma-warpspec';
  if (family === 'mma') return 'cpasync-mma';
  return 'wmma-direct';
}

interface FamilyPaths {
  nodes: NodeId[];
  edges: Edge[];
}

// Variant modifiers (plan §D3). Each true flag adds extra nodes / edges to
// the base family shape without re-writing the skeleton.
interface VariantExtras {
  rs: boolean;
  ts: boolean;
  ws: boolean;
  cg2: boolean;
  sparse: boolean;
  blockScaled: boolean;
}

function pathsForFamily(shape: FamilyShape, extras: VariantExtras): FamilyPaths {
  switch (shape) {
    case 'tma-warpspec': {
      // Base SS warpspec path.
      const nodes: NodeId[] = ['gmemA', 'gmemB', 'gmemC', 'smemA', 'smemB', 'smemC', 'mma', 'acc'];
      const edges: Edge[] = [
        { id: 'tmaA', from: 'gmemA', to: 'smemA', label: 'tma.load' },
        { id: 'tmaB', from: 'gmemB', to: 'smemB', label: 'tma.load' },
        { id: 'descB', from: 'smemB', to: 'mma', label: 'desc' },
        { id: 'mmaOut', from: 'mma', to: 'acc', label: '+=' },
        { id: 'tmemLd', from: 'acc', to: 'smemC', label: 'stmatrix' },
        { id: 'stgSmem', from: 'smemC', to: 'gmemC', label: 'tma.store' },
      ];

      // A operand path depends on variant:
      //   SS → SMEM-A → MMA (desc)
      //   RS → SMEM-A → regA (ldmatrix) → MMA
      //   TS → SMEM-A → TMEM-A (tcgen05.cp) → MMA (desc)
      if (extras.ts) {
        nodes.push('tmemA');
        edges.push({ id: 'tcgen05cp', from: 'smemA', to: 'tmemA', label: 'tcgen05.cp' });
        edges.push({ id: 'descA', from: 'tmemA', to: 'mma', label: 'desc (TS)' });
      } else if (extras.rs) {
        nodes.push('regA');
        edges.push({ id: 'ldA', from: 'smemA', to: 'regA', label: 'ldmatrix' });
        edges.push({ id: 'descA', from: 'regA', to: 'mma', label: 'reg (RS)' });
      } else {
        edges.push({ id: 'descA', from: 'smemA', to: 'mma', label: 'desc' });
      }

      // sparse: metadata tensor ships alongside A+B.
      if (extras.sparse) {
        nodes.push('gmemMeta', 'smemMeta');
        edges.push({ id: 'metaLoad', from: 'gmemMeta', to: 'smemMeta', label: 'tma.meta' });
        edges.push({ id: 'metaDesc', from: 'smemMeta', to: 'mma', label: 'meta' });
      }

      // block_scaled: SFA + SFB TMAs.
      if (extras.blockScaled) {
        nodes.push('gmemSFA', 'smemSFA', 'gmemSFB', 'smemSFB');
        edges.push({ id: 'sfaLoad', from: 'gmemSFA', to: 'smemSFA', label: 'tma.SFA' });
        edges.push({ id: 'sfaDesc', from: 'smemSFA', to: 'mma', label: 'SFA' });
        edges.push({ id: 'sfbLoad', from: 'gmemSFB', to: 'smemSFB', label: 'tma.SFB' });
        edges.push({ id: 'sfbDesc', from: 'smemSFB', to: 'mma', label: 'SFB' });
      }

      return { nodes, edges };
    }
    case 'cpasync-mma':
      return {
        nodes: ['gmemA', 'gmemB', 'gmemC', 'smemA', 'smemB', 'regA', 'regB', 'mma', 'acc'],
        edges: [
          { id: 'tmaA', from: 'gmemA', to: 'smemA', label: 'cp.async' },
          { id: 'tmaB', from: 'gmemB', to: 'smemB', label: 'cp.async' },
          { id: 'ldA', from: 'smemA', to: 'regA', label: 'ldmatrix' },
          { id: 'ldB', from: 'smemB', to: 'regB', label: 'ldmatrix' },
          { id: 'descA', from: 'regA', to: 'mma', label: '' },
          { id: 'descB', from: 'regB', to: 'mma', label: '' },
          { id: 'mmaOut', from: 'mma', to: 'acc', label: 'mma.sync' },
          { id: 'stgSmem', from: 'acc', to: 'gmemC', label: 'stg / st.global' },
        ],
      };
    case 'wmma-direct':
      return {
        // No async copy, no SMEM indirection in the basic wmma path.
        nodes: ['gmemA', 'gmemB', 'gmemC', 'regA', 'regB', 'mma', 'acc'],
        edges: [
          { id: 'ldA', from: 'gmemA', to: 'regA', label: 'wmma.load_matrix_sync' },
          { id: 'ldB', from: 'gmemB', to: 'regB', label: 'wmma.load_matrix_sync' },
          { id: 'descA', from: 'regA', to: 'mma', label: '' },
          { id: 'descB', from: 'regB', to: 'mma', label: '' },
          { id: 'mmaOut', from: 'mma', to: 'acc', label: 'wmma.mma.sync' },
          { id: 'stgSmem', from: 'acc', to: 'gmemC', label: 'wmma.store_matrix_sync' },
        ],
      };
  }
}

function nodeCenter(n: Node) {
  return { cx: n.x + n.w / 2, cy: n.y + n.h / 2, right: n.x + n.w, left: n.x, bottom: n.y + n.h, top: n.y };
}

function edgePath(from: Node, to: Node): string {
  const a = nodeCenter(from);
  const b = nodeCenter(to);
  const forward = a.cx < b.cx;
  const ax = forward ? a.right : a.left;
  const bx = forward ? b.left : b.right;
  if (Math.abs(a.cy - b.cy) < 4) {
    return `M ${ax} ${a.cy} L ${bx} ${b.cy}`;
  }
  // Dogleg: step half-way, then vertical, then rest.
  const mid = (ax + bx) / 2;
  return `M ${ax} ${a.cy} L ${mid} ${a.cy} L ${mid} ${b.cy} L ${bx} ${b.cy}`;
}

function edgePulsePoint(from: Node, to: Node, progress: number) {
  const a = nodeCenter(from);
  const b = nodeCenter(to);
  const forward = a.cx < b.cx;
  const ax = forward ? a.right : a.left;
  const bx = forward ? b.left : b.right;
  return { x: ax + (bx - ax) * progress, y: a.cy + (b.cy - a.cy) * progress };
}

// What's active right now, by stream. Under warpspec multiple streams can be
// bright simultaneously; under coupled only one can be bright at a time.
interface ActiveState {
  producerKind: string | undefined;
  producerSub: string | undefined;       // phase 6: disambiguate tma/meta/scale/etc
  consumerKind: string | undefined;
  epilogueKind: string | undefined;
  hasTmem: boolean;
}

function nodeActive(id: NodeId, s: ActiveState): boolean {
  const hits: boolean[] = [];
  if (s.producerKind === 'tma.load' || s.producerKind === 'cp.async') {
    hits.push(id === 'gmemA' || id === 'gmemB' || id === 'smemA' || id === 'smemB');
  }
  if (s.producerKind === 'ldmatrix' || s.consumerKind === 'ldmatrix') {
    hits.push(id === 'smemA' || id === 'smemB' || id === 'regA' || id === 'regB');
  }
  // Phase 6: TS tcgen05.cp — SMEM-A and TMEM-A light up.
  if (s.producerSub === 'tcgen05-cp') {
    hits.push(id === 'smemA' || id === 'tmemA');
  }
  // Phase 6: sparse metadata TMA.
  if (s.producerSub === 'metadata') {
    hits.push(id === 'gmemMeta' || id === 'smemMeta');
  }
  // Phase 6: block_scaled scale TMA lights all 4 scale nodes.
  if (s.producerSub === 'scale') {
    hits.push(id === 'gmemSFA' || id === 'smemSFA' || id === 'gmemSFB' || id === 'smemSFB');
  }
  if (s.consumerKind === 'wgmma.step' || s.consumerKind === 'tcgen05.mma.step') {
    hits.push(id === 'smemA' || id === 'smemB' || id === 'regA' || id === 'regB'
          || id === 'mma' || id === 'acc'
          // If tmemA is in the path (TS variant), it's live during consumer.
          || id === 'tmemA');
  }
  if (s.epilogueKind === 'tcgen05.ld') {
    hits.push(s.hasTmem && (id === 'acc' || id === 'regA' || id === 'regB'));
  }
  if (s.epilogueKind === 'epilogue.stg_smem') {
    hits.push(id === 'acc' || id === 'smemC');
  }
  if (s.epilogueKind === 'epilogue.tma.store') {
    hits.push(id === 'smemC' || id === 'gmemC');
  }
  return hits.some(Boolean);
}

function edgeActive(id: EdgeId, s: ActiveState): boolean {
  const hits: boolean[] = [];
  if (s.producerKind === 'tma.load' || s.producerKind === 'cp.async') {
    // For wmma there are no tmaA/tmaB edges — ldA/ldB carry the GMEM→reg load.
    hits.push(id === 'tmaA' || id === 'tmaB' || id === 'ldA' || id === 'ldB');
  }
  if (s.producerKind === 'ldmatrix' || s.consumerKind === 'ldmatrix') {
    hits.push(id === 'ldA' || id === 'ldB');
  }
  // Phase 6: TS tcgen05.cp edge.
  if (s.producerSub === 'tcgen05-cp') {
    hits.push(id === 'tcgen05cp');
  }
  // Phase 6: sparse metadata load edge.
  if (s.producerSub === 'metadata') {
    hits.push(id === 'metaLoad');
  }
  // Phase 6: block_scaled SFA+SFB load edges.
  if (s.producerSub === 'scale') {
    hits.push(id === 'sfaLoad' || id === 'sfbLoad');
  }
  if (s.consumerKind === 'wgmma.step' || s.consumerKind === 'tcgen05.mma.step') {
    // Descriptor/reg edges into MMA, plus auxiliary desc edges when variant present.
    hits.push(id === 'descA' || id === 'descB' || id === 'mmaOut'
          || id === 'metaDesc' || id === 'sfaDesc' || id === 'sfbDesc');
  }
  if (s.epilogueKind === 'tcgen05.ld') {
    hits.push(s.hasTmem && id === 'tmemLd');
  }
  if (s.epilogueKind === 'epilogue.stg_smem') {
    // For tcgen05/wgmma: stmatrix is an intermediate (acc→smemC). For coupled
    // sm_80 mma and sm_70 wmma: the single epilogue edge (acc→gmemC) fires.
    hits.push(id === 'tmemLd' || id === 'stgSmem');
  }
  if (s.epilogueKind === 'epilogue.tma.store') {
    hits.push(id === 'stgSmem');
  }
  return hits.some(Boolean);
}

// Pick the progress-driving phase for each edge so the pulse circle moves
// along the right stream.
function edgeStreamProgress(
  id: EdgeId,
  producerProg: number,
  consumerProg: number,
  epilogueProg: number,
): number {
  if (id === 'tmaA' || id === 'tmaB') return producerProg;
  if (id === 'ldA' || id === 'ldB') return producerProg; // coupled-mode ldmatrix rides producer stream
  if (id === 'descA' || id === 'descB' || id === 'mmaOut') return consumerProg;
  // Phase 6 — producer sub-phases ride the producer stream; desc edges
  // for auxiliary operands ride the consumer stream.
  if (id === 'tcgen05cp') return producerProg;
  if (id === 'metaLoad' || id === 'sfaLoad' || id === 'sfbLoad') return producerProg;
  if (id === 'metaDesc' || id === 'sfaDesc' || id === 'sfbDesc') return consumerProg;
  if (id === 'tmemLd' || id === 'stgSmem') return epilogueProg;
  return 0;
}

const SVG_W = 760;
const SVG_H = 290;

export function MemFlowPanel() {
  const i = inst.value;
  const cProd = currentProducerPhase.value;
  const cCons = currentConsumerPhase.value;
  const cEpi = currentEpiloguePhase.value;
  // Per-stream progress reads from world.value (plan §D3): pulse speed per
  // edge follows its own stream, so TMA edges animate even while MMA edges
  // are also pulsing. Consumer uses world.progress.consumer directly rather
  // than the precedence-based `phaseProgress` so the value is stable even
  // when another stream is active.
  const pProg = producerPhaseProgress.value;
  const cProg = world.value.progress.consumer;
  const eProg = epiloguePhaseProgress.value;
  const mode = pipelineMode.value;
  const hasTmem = i.accIn === 'tmem' || i.aSource.includes('tmem');

  const active: ActiveState = {
    producerKind: cProd?.kind,
    producerSub: cProd?.producerSub,
    consumerKind: cCons?.kind,
    epilogueKind: cEpi?.kind,
    hasTmem,
  };

  // Customize acc sublabel based on whether acc lives in TMEM or .reg.
  const accSub = hasTmem ? 'TMEM' : '.reg';

  // Header summary — show whether producer & consumer are both live (warpspec
  // steady state) so the reader sees the overlap in this view too.
  const bothLive = mode === 'warpspec' && cProd && cCons;

  // Pick the family-specific subset of nodes and edges. wmma does not
  // traverse SMEM at all; sm_80 mma traverses SMEM but skips the SMEM-C
  // staging in the epilogue; sm_90/100 use the full TMA-backed path.
  const shape = familyShape(i.family);
  const sum = summary.value;
  const extras: VariantExtras = {
    rs: sum.variant === 'rs',
    ts: sum.variant === 'ts',
    ws: sum.extras.warpSpecialized,
    cg2: sum.extras.ctaGroup === 2,
    sparse: sum.extras.sparse,
    blockScaled: sum.extras.blockScaled,
  };
  const paths = pathsForFamily(shape, extras);
  const visibleNodeIds = new Set<NodeId>(paths.nodes);
  const edges = paths.edges;

  // Concise path summary per family.
  const pathSummary =
    shape === 'tma-warpspec'
      ? 'GMEM → SMEM → MMA → acc → SMEM → GMEM (TMA + descriptor loads, stmatrix staging).'
      : shape === 'cpasync-mma'
        ? 'GMEM → SMEM → .reg → MMA → acc → GMEM (cp.async + ldmatrix, no SMEM epilogue staging).'
        : 'GMEM → .reg → MMA → acc → GMEM (wmma.load/store_matrix_sync, no SMEM).';

  return (
    <div class="panel mf">
      <h3>
        Memory flow
        <small>
          — {pathSummary} Active node/arrow highlights follow the Timeline.
        </small>
      </h3>

      <svg class="mf__svg" viewBox={`0 0 ${SVG_W} ${SVG_H}`} width="100%">
        {/* Column bands for visual grouping */}
        <rect x={0}   y={0} width={128} height={SVG_H} class="mf__col mf__col--gmem" />
        <rect x={150} y={0} width={128} height={SVG_H} class="mf__col mf__col--smem" />
        <rect x={300} y={0} width={128} height={SVG_H} class="mf__col mf__col--reg" />
        <rect x={450} y={0} width={128} height={SVG_H} class="mf__col mf__col--mma" />
        <rect x={600} y={0} width={128} height={SVG_H} class="mf__col mf__col--acc" />

        {/* Column labels */}
        <text x={64}  y={SVG_H - 6} text-anchor="middle" class="mf__collbl">GMEM (global)</text>
        <text x={214} y={SVG_H - 6} text-anchor="middle" class="mf__collbl">SMEM</text>
        <text x={364} y={SVG_H - 6} text-anchor="middle" class="mf__collbl">.reg (LRF)</text>
        <text x={514} y={SVG_H - 6} text-anchor="middle" class="mf__collbl">tensor core</text>
        <text x={664} y={SVG_H - 6} text-anchor="middle" class="mf__collbl">{accSub}</text>

        {/* Edges first so nodes sit on top */}
        {edges.map((e) => {
          const from = NODES.find((n) => n.id === e.from)!;
          const to = NODES.find((n) => n.id === e.to)!;
          const isActive = edgeActive(e.id, active);
          const path = edgePath(from, to);
          const a = nodeCenter(from);
          const b = nodeCenter(to);
          const forward = a.cx < b.cx;
          const ax = forward ? a.right : a.left;
          const bx = forward ? b.left : b.right;
          const midx = (ax + bx) / 2;
          const labelX = Math.abs(a.cy - b.cy) < 4 ? midx : midx + (forward ? 4 : -4);
          const labelY = Math.abs(a.cy - b.cy) < 4 ? a.cy - 6 : (a.cy + b.cy) / 2;
          const labelText =
            e.id === 'tmemLd' && hasTmem ? 'tcgen05.ld' : e.label;
          const streamProg = edgeStreamProgress(e.id, pProg, cProg, eProg);
          return (
            <g key={e.id} class={`mf__edge ${isActive ? 'is-active' : ''}`}>
              <path d={path} class="mf__edge-path" marker-end="url(#mf-arrow)" />
              {isActive && (() => {
                const pt = edgePulsePoint(from, to, streamProg);
                return <circle class="mf__edge-pulse" cx={pt.x} cy={pt.y} r={4} />;
              })()}
              {labelText && (
                <text x={labelX} y={labelY} text-anchor="middle" class={`mf__edge-lbl ${isActive ? 'is-active' : ''}`}>
                  {labelText}
                </text>
              )}
            </g>
          );
        })}

        {/* Arrow-head marker */}
        <defs>
          <marker id="mf-arrow" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="8" markerHeight="8" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" class="mf__arrowhead" />
          </marker>
        </defs>

        {/* Nodes (only those relevant for this family's data path) */}
        {NODES.filter((n) => visibleNodeIds.has(n.id)).map((n) => {
          const isActive = nodeActive(n.id, active);
          const sub = n.id === 'acc' ? accSub : n.sub;
          return (
            <g key={n.id} class={`mf__node ${n.cls} ${isActive ? 'is-active' : ''}`}>
              <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={6} />
              <text x={n.x + n.w / 2} y={n.y + 18} text-anchor="middle" class="mf__node-lbl">
                {n.label}
              </text>
              {sub && (
                <text x={n.x + n.w / 2} y={n.y + 34} text-anchor="middle" class="mf__node-sub">
                  {sub}
                </text>
              )}
            </g>
          );
        })}

        {/* Stream status line (top-right corner). Shows every live stream so
            the warpspec overlap is visible even before you look at the
            Timeline. */}
        <text x={SVG_W - 8} y={14} text-anchor="end" class="mf__phaselbl">
          {bothLive ? (
            <>
              producer <tspan class="mf__phaselbl--prod">{cProd!.kind}</tspan>
              {' ∥ '}
              consumer <tspan class="mf__phaselbl--cons">{cCons!.kind}</tspan>
            </>
          ) : cProd ? (
            <>producer {cProd.kind} · {Math.round(pProg * 100)}%</>
          ) : cCons ? (
            <>consumer {cCons.kind} · {Math.round(cProg * 100)}%</>
          ) : cEpi ? (
            <>epilogue {cEpi.kind} · {Math.round(eProg * 100)}%</>
          ) : (
            '— idle —'
          )}
        </text>
      </svg>

      <TruthFooter
        models="per-STREAM active data path across GMEM / SMEM / TMEM / .reg / tensor core / acc; producer and consumer edges light simultaneously when their phases overlap in time (warpspec); edge pulse speed is per-stream progress from world; family-specific nodes and edges (tmemA for TS, gmemMeta/smemMeta for sparse, gmemSFA/SFB for block_scaled, regA for RS)."
        schematic="edge pulse animation speed does not represent real transfer latency; ldmatrix and descriptor-read arrivals are modelled as instantaneous at phase boundaries."
        cite="simulation.ts · world.active / world.progress; CUTLASS CollectiveMma load / mma / store stages"
      />
    </div>
  );
}
