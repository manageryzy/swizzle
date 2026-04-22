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
