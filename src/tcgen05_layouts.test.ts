import { describe, expect, it } from 'vitest';
import { PTX_LAYOUTS, classifyLayout, peerMRange } from './tcgen05_layouts';

describe('PTX tcgen05 layouts A–G', () => {
  it('M=256 cta_group::2 → Layout A', () => {
    expect(classifyLayout(256, 2)).toBe('A');
  });
  it('M=128 cta_group::2 dense → Layout B', () => {
    expect(classifyLayout(128, 2)).toBe('B');
  });
  it('M=128 cta_group::2 sparse → Layout C', () => {
    expect(classifyLayout(128, 2, { sparse: true })).toBe('C');
  });
  it('M=128 cta_group::1 → Layout D', () => {
    expect(classifyLayout(128, 1)).toBe('D');
  });
  it('M=64 cta_group::1 ws → Layout E', () => {
    expect(classifyLayout(64, 1, { warpSpecialized: true })).toBe('E');
  });
  it('M=64 cta_group::1 non-ws → Layout F', () => {
    expect(classifyLayout(64, 1)).toBe('F');
  });
  it('M=32 cta_group::1 .ws → Layout G', () => {
    expect(classifyLayout(32, 1, { warpSpecialized: true })).toBe('G');
  });
  it('M=32 cta_group::1 non-ws is invalid (PTX Table 41)', () => {
    expect(classifyLayout(32, 1)).toBeNull();
  });
  it('invalid combinations return null', () => {
    expect(classifyLayout(64, 2)).toBeNull();
    expect(classifyLayout(256, 1)).toBeNull();
  });
  it('every layout has a description', () => {
    for (const l of Object.values(PTX_LAYOUTS)) {
      expect(l.description.length).toBeGreaterThan(30);
    }
  });

  it('peerMRange splits 2-CTA accumulators evenly', () => {
    expect(peerMRange('A', 0, 256)).toMatchObject({ lo: 0, hi: 128 });
    expect(peerMRange('A', 1, 256)).toMatchObject({ lo: 128, hi: 256 });
    expect(peerMRange('B', 0, 128)).toMatchObject({ lo: 0, hi: 64 });
    expect(peerMRange('B', 1, 128)).toMatchObject({ lo: 64, hi: 128 });
    expect(peerMRange('D', 0, 128)).toMatchObject({ lo: 0, hi: 128, peerLabel: 'single CTA' });
  });
});
