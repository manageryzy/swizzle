import { describe, expect, it } from 'vitest';
import { canonicalLayout, canonicalT } from './gmma_layouts';

describe('canonical GMMA/UMMA SMEM layouts', () => {
  it('sm_90 K-major 128B has Swizzle<3,4,3>', () => {
    const l = canonicalLayout('sm90', 'K', '128B');
    expect(l?.swizzle).toBe('Swizzle<3,4,3>');
    expect(l?.shape).toContain('(8,n)');
  });

  it('sm_90 MN-major 64B has Swizzle<2,4,3>', () => {
    const l = canonicalLayout('sm90', 'MN', '64B');
    expect(l?.swizzle).toBe('Swizzle<2,4,3>');
  });

  it('sm_100 MN-major 128B_BASE32B is only valid on sm_100', () => {
    expect(canonicalLayout('sm100', 'MN', '128B.base32B')?.swizzle).toBe('Swizzle<2,5,2>');
    expect(canonicalLayout('sm90', 'MN', '128B.base32B')).toBeNull();
  });

  it('fp16 has T=8, fp32 has T=4', () => {
    expect(canonicalT(2)).toBe(8); // fp16
    expect(canonicalT(4)).toBe(4); // fp32
    expect(canonicalT(1)).toBe(16); // fp8
  });
});
