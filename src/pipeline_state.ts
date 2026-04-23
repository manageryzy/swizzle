// Pipeline state tracking — which K slice each SMEM ring stage holds at a
// given mma iteration. Mirrors CUTLASS `PipelineState` semantics from
//   include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
//
// Key facts:
//   * SMEM tensor is shaped `(BLK_M, BLK_K, PIPE=kStages)`.
//   * Producer warpgroup: `producer_acquire(w) → tma_load(...sA(_,_,_,w)) → ++w`
//   * Consumer warpgroup: `consumer_wait(r) → gemm(...sA(_,_,_,r)) → ++r`
//   * Both indices wrap modulo `kStages`; producer is ahead of consumer by
//     up to `kStages - 1` iterations in warp-specialized mode.
//
// Steady-state ring state at consumer iter X:
//   consumer_stage = X mod kStages           (reading slice X)
//   producer_stage = (X + kStages - 1) mod kStages   (writing slice X + kStages - 1)
//
// Every stage `s` holds the K slice that was most recently written into it:
//   stageHoldsSlice(s) = (X + kStages - 1) - ((X + kStages - 1 - s) mod kStages)
// which yields slice indices {X, X+1, X+2, ..., X+kStages-1} distributed
// across the stage slots in ring order.

import type { InstSpec, PipelineMode } from './instructions';

export type { PipelineMode };

// One phase on the emitted timeline. Shape is identical to `Phase` in
// `state.ts` but we keep a local type here so emitTimeline can live in pure
// module land without importing signals. `iter` is the K iteration the phase
// corresponds to (useful for cross-linking with ringState).
export interface TimelinePhase {
  id: string;
  kind:
    | 'tma.load'
    | 'cp.async'
    | 'ldmatrix'
    | 'wmma.load'
    | 'tcgen05.cp'
    | 'metadata'
    | 'scale'
    | 'wgmma.step'
    | 'tcgen05.mma.step'
    | 'tcgen05.ld'
    | 'tcgen05.st'
    | 'epilogue.stg_smem'
    | 'epilogue.tma.store';
  startTick: number;
  endTick: number;
  label: string;
  description: string;
  iter?: number;
  stage?: number;
  /** K-slab index (producer: which TMA; consumer: which slab of atomsPerStage_K). */
  kSlab?: number;
  /** For consumer phases: atom-K slice within the slab (0..atomsPerStage_K). */
  kAtomInSlab?: number;
  /** For collapse phases: how many additional iters this phase represents. */
  collapsedCount?: number;
  /** Producer sub-phase classification (used by Phase 3 MemFlow wiring). */
  producerSub?: 'tma' | 'cpasync' | 'wmma-load' | 'ldmatrixA' | 'tcgen05-cp' | 'metadata' | 'scale';
}

export interface StageState {
  stage: number;         // stage slot index [0, kStages)
  slice: number;         // K-slice number currently held (0-indexed into outer K)
  role: 'consume' | 'fill' | 'hold';
}

export function ringState(iter: number, kStages: number): StageState[] {
  const consumer = iter % kStages;
  const producer = (iter + kStages - 1) % kStages;
  const states: StageState[] = [];
  const head = iter + kStages - 1; // producer's current slice
  for (let s = 0; s < kStages; s++) {
    const slice = head - ((head - s + kStages) % kStages);
    const role: StageState['role'] =
      s === consumer ? 'consume' : s === producer ? 'fill' : 'hold';
    states.push({ stage: s, slice, role });
  }
  return states;
}

// Timeline annotation spans — what the Timeline panel draws as shaded stalls
// and mbarrier markers. This is a schematic derivation from (numIters, kStages),
// NOT a cycle simulator: we know that in warp-specialized mode the consumer
// cannot start until stage 0 is full, and at the tail of the K loop the
// producer idles while the consumer drains the last (kStages-1) stages.
//
// Inputs are indices into the schematic timeline: `producerPhaseTicks` is the
// duration (in ticks) of one TMA/cp.async phase; `consumerPhaseTicks` is one
// mma.step phase; `firstConsumerTick` is when the consumer actually starts.
// Outputs are intervals `[startTick, endTick)` measured in the same ticks.
export interface StallSpan {
  kind: 'consumer-wait-fill' | 'producer-wait-empty' | 'mbarrier-arrive' | 'mbarrier-wait';
  startTick: number;
  endTick: number;
  stage?: number;
  label: string;
}

export interface PipelineTimelineInput {
  numIters: number;
  kStages: number;
  /** Tick at which the first TMA/cp.async phase begins (usually 0). */
  producerStartTick: number;
  /** Duration of one TMA/cp.async phase. */
  producerPhaseTicks: number;
  /** Tick at which the first mma.step begins. */
  firstConsumerTick: number;
  /** Duration of one mma.step. */
  consumerPhaseTicks: number;
}

export function pipelineAnnotations(input: PipelineTimelineInput): StallSpan[] {
  const {
    numIters,
    kStages,
    producerStartTick,
    producerPhaseTicks,
    firstConsumerTick,
    consumerPhaseTicks,
  } = input;
  const spans: StallSpan[] = [];
  if (numIters === 0 || kStages === 0) return spans;

  // Consumer warmup: before the first mma.step the consumer has been blocked
  // on `mbar[0].full`. Shade from producerStartTick to firstConsumerTick on
  // the consumer lane.
  if (firstConsumerTick > producerStartTick) {
    spans.push({
      kind: 'consumer-wait-fill',
      startTick: producerStartTick,
      endTick: firstConsumerTick,
      stage: 0,
      label: `consumer waits on mbar[0].full (warmup, ${kStages}-deep ring)`,
    });
  }

  // Mbarrier arrive markers: one per producer phase completion, up to kStages
  // loads (we only schematically render the first load; repeated fills happen
  // at each consumer step in steady state, which we surface as dependency
  // markers at the consumer lane).
  for (let s = 0; s < Math.min(kStages, numIters); s++) {
    const t = producerStartTick + producerPhaseTicks * Math.min(s + 1, 1);
    spans.push({
      kind: 'mbarrier-arrive',
      startTick: t,
      endTick: t,
      stage: s,
      label: `producer arrive mbar[${s}].full`,
    });
  }

  // Mbarrier wait markers: consumer acquires the full stage at the start of
  // each mma.step (schematically: stages cycle iter % kStages).
  for (let k = 0; k < numIters; k++) {
    const t = firstConsumerTick + consumerPhaseTicks * k;
    spans.push({
      kind: 'mbarrier-wait',
      startTick: t,
      endTick: t,
      stage: k % kStages,
      label: `consumer wait mbar[${k % kStages}].full`,
    });
  }

  // Producer idle tail: producer has nothing to fill once all K slices are
  // issued, so the last (kStages - 1) consumer iters run with an empty
  // producer. Shade on the producer lane.
  if (numIters > kStages - 1) {
    const tailStart = firstConsumerTick + consumerPhaseTicks * (numIters - kStages + 1);
    const tailEnd = firstConsumerTick + consumerPhaseTicks * numIters;
    spans.push({
      kind: 'producer-wait-empty',
      startTick: tailStart,
      endTick: tailEnd,
      label: `producer idle — last ${kStages - 1} stages drained by consumer`,
    });
  }

  return spans;
}

// ---------------------------------------------------------------------------
// Timeline emission
// ---------------------------------------------------------------------------
//
// `emitTimeline` returns three phase streams whose ticks may overlap when
// `pipelineMode === 'warpspec'`. The shape mirrors CUTLASS
// `MainloopSm90TmaGmmaWarpSpecialized` (producer warpgroup issues TMA while
// consumer warpgroup executes WGMMA under an mbarrier-coordinated ring). For
// `'coupled'` we collapse back to a single serial lane (sm_70 wmma / sm_80 mma).
//
// Tick accounting — schematic, not cycle-level:
//   TMA_TICKS = 4, MMA_TICKS = 2, LDM_TICKS = 2, STG_TICKS = 2, TMA_ST_TICKS = 2
// These tick budgets are identical to the ones used by the pre-v3 `phasesFor`
// so that panels which read phase durations see no visible change in default
// layout.

const TMA_TICKS = 4;
const CPASYNC_TICKS = 4;
const MMA_TICKS = 2;
const LDM_TICKS = 2;
const TCG_LD_TICKS = 3;
const STG_SMEM_TICKS = 2;
const TMA_STORE_TICKS = 2;

// Cap on individually-rendered consumer/producer iterations. Beyond this,
// the remaining K iterations collapse into a single "×N more" bar to keep
// the timeline legible on the CTA-level kernel loops (tileK/atomK=16, 32, …).
const MAX_PHASES_PER_STREAM = 8;

export interface TimelineStreams {
  producer: TimelinePhase[];
  consumer: TimelinePhase[];
  epilogue: TimelinePhase[];
  totalTicks: number;
}

function mmaStepKind(i: InstSpec): TimelinePhase['kind'] {
  return i.family === 'tcgen05' || i.family === 'tcgen05.block_scaled'
    ? 'tcgen05.mma.step'
    : 'wgmma.step';
}

function producerKindFor(i: InstSpec, mode: PipelineMode): TimelinePhase['kind'] {
  // sm_90 wgmma / sm_100 tcgen05 use cp.async.bulk.tensor (TMA).
  // sm_80 mma uses plain cp.async → SMEM + ldmatrix → .reg.
  // sm_70/72/75 wmma uses a warp-wide load_matrix_sync directly from
  // gmem/shared — no async copy at all. We still need a producer phase to
  // represent that load on the timeline; reuse the 'ldmatrix' kind because
  // its semantic ("load fragment into registers") is the closest match.
  if (mode === 'coupled') {
    if (i.family === 'wmma') return 'ldmatrix';
    return 'cp.async';
  }
  return 'tma.load';
}

/**
 * Emit three phase streams (producer / consumer / epilogue) with tick ranges
 * that reflect the pipeline regime.
 *
 * Warpspec: producer emits `numIters` loads back-to-back starting at t=0;
 * consumer's iter i starts at max(prev_consumer_end, producer_end_of(iter=i)).
 * That yields the (kStages−1)-ahead steady-state once the ring is primed.
 *
 * Coupled: for each iter, producer load (TMA/cp.async) runs, then consumer
 * mma step runs, strictly serial; epilogue appended after.
 */
export function emitTimeline(
  i: InstSpec,
  numIters: number,
  kStages: number,
  mode: PipelineMode,
): TimelineStreams {
  const producer: TimelinePhase[] = [];
  const consumer: TimelinePhase[] = [];
  const epilogue: TimelinePhase[] = [];

  const pKind = producerKindFor(i, mode);
  const cKind = mmaStepKind(i);
  const accInLbl = i.accIn === 'rmem' ? '.reg' : i.accIn.toUpperCase();
  const pTicks = pKind === 'tma.load' ? TMA_TICKS : CPASYNC_TICKS;

  // Cap individually-rendered phases so the timeline stays legible.
  const displayedIters = Math.min(numIters, MAX_PHASES_PER_STREAM);
  const tailIters = numIters - displayedIters;

  // Helper: "iter i's producer end tick" — used to gate the consumer. In
  // steady state the consumer consumes `(i % kStages)` as soon as the
  // producer finishes writing it.
  const producerEndForIter = (iter: number): number => {
    // Each producer slot is pTicks long and dispatched in iter order, but a
    // producer can only write stage s = iter % kStages once the consumer has
    // drained it. For the head kStages iters there's no back-pressure
    // (nothing to consume yet) so they pack back-to-back.
    return producer[iter]?.endTick ?? 0;
  };

  if (mode === 'warpspec') {
    // Seed producer for ALL iters first; we'll adjust for back-pressure below.
    let pCursor = 0;
    for (let k = 0; k < displayedIters; k++) {
      const stage = k % kStages;
      // Back-pressure: producer for iter k can only start once consumer has
      // finished iter (k - kStages), which freed stage (stage). That end tick
      // is set below after the consumer loop; for the initial emit we pack
      // eagerly (overestimate = ok, fixup step corrects).
      const start = pCursor;
      const end = start + pTicks;
      producer.push({
        id: `${i.id}.${pKind}.prod.${k}`,
        kind: pKind,
        startTick: start,
        endTick: end,
        label: k === 0 ? `TMA load stage ${stage} (prime ring)` : `TMA load stage ${stage} · k=${k}`,
        description: `Producer warpgroup issues ${pKind} for K slice ${k} into ring stage ${stage}. Arrives on mbar[${stage}].full when complete.`,
        iter: k,
        stage,
      });
      pCursor = end;
    }
    if (tailIters > 0) {
      const start = pCursor;
      const end = start + pTicks;
      producer.push({
        id: `${i.id}.${pKind}.prod.tail`,
        kind: pKind,
        startTick: start,
        endTick: end,
        label: `TMA × ${tailIters} more`,
        description: `${tailIters} additional TMA loads collapsed for legibility; each fills stage (k mod ${kStages}) on the ring.`,
        iter: displayedIters,
      });
    }

    // Consumer starts once stage 0 is primed (the k=0 producer has finished).
    // From iter 1 onward, each iter starts at max(prev end, producer end for that iter).
    let cCursor = producer[0]?.endTick ?? 0;
    for (let k = 0; k < displayedIters; k++) {
      const stage = k % kStages;
      const producerReady = producerEndForIter(k);
      const start = Math.max(cCursor, producerReady);
      const end = start + MMA_TICKS;
      consumer.push({
        id: `${i.id}.${cKind}.cons.${k}`,
        kind: cKind,
        startTick: start,
        endTick: end,
        label: `mma iter ${k + 1}/${numIters} · stage ${stage}`,
        description: `Tensor core consumes K atom ${k} from stage ${stage}; accumulator in ${accInLbl}. Producer may already be filling stage ${(k + kStages - 1) % kStages}.`,
        iter: k,
        stage,
      });
      cCursor = end;
    }
    if (tailIters > 0) {
      const start = cCursor;
      const end = start + MMA_TICKS;
      consumer.push({
        id: `${i.id}.${cKind}.cons.tail`,
        kind: cKind,
        startTick: start,
        endTick: end,
        label: `mma × ${tailIters} more`,
        description: `${tailIters} additional K iterations collapsed; each consumes one stage of the ring and lands in ${accInLbl}.`,
        iter: displayedIters,
      });
    }

    // Fixup: if a producer iter would have finished before the consumer
    // cleared the ring slot it needs, push the producer phase forward so it
    // doesn't overlap the consumer on that stage. This surfaces as a
    // visually-stalled producer bar when the ring is full.
    for (let k = kStages; k < producer.length; k++) {
      const consumerIdxThatFreedStage = k - kStages;
      const consumerEnd = consumer[consumerIdxThatFreedStage]?.endTick ?? 0;
      if (producer[k].startTick < consumerEnd) {
        const delta = consumerEnd - producer[k].startTick;
        producer[k].startTick += delta;
        producer[k].endTick += delta;
        // Any subsequent producer tasks that now overlap with a pushed-forward
        // predecessor also need to slide.
        for (let j = k + 1; j < producer.length; j++) {
          if (producer[j].startTick < producer[j - 1].endTick) {
            const d = producer[j - 1].endTick - producer[j].startTick;
            producer[j].startTick += d;
            producer[j].endTick += d;
          }
        }
      }
    }
  } else {
    // Coupled: same warp does producer → consumer → producer → consumer. Two
    // subshapes:
    //   sm_80 mma: cp.async → SMEM, ldmatrix → .reg, mma.sync.
    //   sm_70 wmma: wmma.load_matrix_sync directly from gmem/shared into the
    //               warp-wide fragment — no cp.async, no SMEM staging, no
    //               ldmatrix.
    const isWmma = i.family === 'wmma';
    let t = 0;
    for (let k = 0; k < displayedIters; k++) {
      const prodLabel = isWmma
        ? (k === 0 ? 'wmma.load_matrix_sync A,B' : `load iter ${k}`)
        : (k === 0 ? 'cp.async → SMEM (A,B)' : `cp.async iter ${k}`);
      const prodDesc = isWmma
        ? `Warp-wide wmma.load_matrix_sync fetches A and B fragments for K slice ${k} from gmem or shared (no async copy, no SMEM staging).`
        : `Per-thread cp.async load for K slice ${k}. Same warp stalls on cp.async.wait_group before ldmatrix / mma.`;
      producer.push({
        id: `${i.id}.${pKind}.prod.${k}`,
        kind: pKind,
        startTick: t,
        endTick: t + pTicks,
        label: prodLabel,
        description: prodDesc,
        iter: k,
        stage: 0,
      });
      t += pTicks;
      if (i.family === 'mma') {
        // Coupled mma has an ldmatrix hop between SMEM load and mma.sync.
        producer.push({
          id: `${i.id}.ldmatrix.prod.${k}`,
          kind: 'ldmatrix',
          startTick: t,
          endTick: t + LDM_TICKS,
          label: `ldmatrix iter ${k}`,
          description: 'Warp loads matrix fragment from swizzled SMEM into per-lane registers.',
          iter: k,
          stage: 0,
        });
        t += LDM_TICKS;
      }
      consumer.push({
        id: `${i.id}.${cKind}.cons.${k}`,
        kind: cKind,
        startTick: t,
        endTick: t + MMA_TICKS,
        label: `mma iter ${k + 1}/${numIters}`,
        description: `Tensor core consumes K atom ${k}; accumulator in ${accInLbl}. Same warp will issue the next load after this completes.`,
        iter: k,
        stage: 0,
      });
      t += MMA_TICKS;
    }
    if (tailIters > 0) {
      producer.push({
        id: `${i.id}.${pKind}.prod.tail`,
        kind: pKind,
        startTick: t,
        endTick: t + pTicks,
        label: `cp.async × ${tailIters} more`,
        description: `${tailIters} additional load-mma pairs collapsed for legibility.`,
        iter: displayedIters,
      });
      t += pTicks;
      consumer.push({
        id: `${i.id}.${cKind}.cons.tail`,
        kind: cKind,
        startTick: t,
        endTick: t + MMA_TICKS,
        label: `mma × ${tailIters} more`,
        description: `${tailIters} additional K iterations collapsed.`,
        iter: displayedIters,
      });
      t += MMA_TICKS;
    }
  }

  // Epilogue: comes after both streams drain. Shape differs by family:
  //   wgmma/tcgen05: [tcgen05.ld only for tcgen05] → stmatrix into SMEM-C →
  //                  cp.async.bulk.tensor.store drains to GMEM.
  //   mma (sm_80):   .reg → gmem directly via st.global (no SMEM staging).
  //   wmma:          wmma.store_matrix_sync writes the fragment straight to
  //                  gmem or shared (one step).
  const mainloopEnd = Math.max(
    producer.at(-1)?.endTick ?? 0,
    consumer.at(-1)?.endTick ?? 0,
  );
  let eCursor = mainloopEnd;
  if (mode === 'warpspec') {
    if (i.accIn === 'tmem') {
      epilogue.push({
        id: `${i.id}.tcgen05.ld`,
        kind: 'tcgen05.ld',
        startTick: eCursor,
        endTick: eCursor + TCG_LD_TICKS,
        label: 'tcgen05.ld → .reg',
        description: 'Move accumulator from TMEM into per-lane registers for epilogue (16dp×N shapes; lanes fan out across subpartitions).',
      });
      eCursor += TCG_LD_TICKS;
    }
    epilogue.push({
      id: `${i.id}.epilogue.stg_smem`,
      kind: 'epilogue.stg_smem',
      startTick: eCursor,
      endTick: eCursor + STG_SMEM_TICKS,
      label: 'acc → SMEM staging',
      description: 'Warp writes its accumulator fragments into an epilogue SMEM region (stmatrix). Fills left-to-right.',
    });
    eCursor += STG_SMEM_TICKS;
    epilogue.push({
      id: `${i.id}.epilogue.tma.store`,
      kind: 'epilogue.tma.store',
      startTick: eCursor,
      endTick: eCursor + TMA_STORE_TICKS,
      label: 'SMEM → GMEM (TMA store)',
      description: 'cp.async.bulk.tensor.store drains the staging region to GMEM. Cells empty right-to-left.',
    });
    eCursor += TMA_STORE_TICKS;
  } else {
    // Coupled: one step, registers straight to GMEM.
    const label = i.family === 'wmma'
      ? 'wmma.store_matrix_sync → GMEM (C)'
      : '.reg → GMEM (st.global)';
    const desc = i.family === 'wmma'
      ? 'Warp-wide wmma.store_matrix_sync writes the C fragment to gmem or shared directly; no SMEM staging.'
      : 'Each thread writes its mma accumulator fragment with plain st.global / stg; no SMEM staging.';
    epilogue.push({
      id: `${i.id}.epilogue.stg_smem`,
      kind: 'epilogue.stg_smem',
      startTick: eCursor,
      endTick: eCursor + TMA_STORE_TICKS,
      label,
      description: desc,
    });
    eCursor += TMA_STORE_TICKS;
  }

  return { producer, consumer, epilogue, totalTicks: eCursor };
}
