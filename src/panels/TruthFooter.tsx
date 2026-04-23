// A single-line disclosure shown under every panel: "models: X / schematic: Y".
// Keeps the visualizer honest about what is derived from verified spec-accurate
// code (the `models` half) vs what is a didactic stand-in (the `schematic`
// half). Hovering expands the detail; no modal.

import type { JSX } from 'preact';

export interface TruthFooterProps {
  models: string;
  schematic: string;
  /** Optional source reference — PTX §, CUTLASS header path, etc. */
  cite?: string;
  /** If true, shows a green "verified vs cute" pill. */
  verified?: boolean;
}

export function TruthFooter({ models, schematic, cite, verified }: TruthFooterProps): JSX.Element {
  return (
    <div class="truth">
      {verified && <span class="truth__verified" title="Cross-checked against cute headers in verify/">verified vs cute</span>}
      <span class="truth__models">
        <span class="truth__lbl">models</span>
        {models}
      </span>
      <span class="truth__sep">·</span>
      <span class="truth__schematic">
        <span class="truth__lbl">schematic</span>
        {schematic}
      </span>
      {cite && <span class="truth__ref" title={cite}>{cite}</span>}
    </div>
  );
}
