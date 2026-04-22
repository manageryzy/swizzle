// verify_standalone.cpp — Tier 3 fallback that does not need CUTLASS / CUDA.
//
// This is a minimal hand-port of the cute::Swizzle and cute::Layout formulas
// (copied from cute/swizzle.hpp and cute/layout.hpp comments), producing the
// same JSON shape as verify.cpp. Running it generates `expected.json`, which
// the TS test suite diffs against.
//
// Compile:
//   g++ -std=c++17 -O0 verify_standalone.cpp -o verify_standalone
//
// If the TS port disagrees with verify_standalone, and verify.cpp (cute)
// agrees with verify_standalone, then TS is wrong. If verify.cpp disagrees,
// the cute upstream has changed — refresh this file too.

#include <cstdio>
#include <cstdint>
#include <vector>

// ---- cute::Swizzle<B,M,S>::apply ------------------------------------------
//   result = offset XOR ((offset & yyy_mask) >> S)
//   yyy_mask = ((1 << B) - 1) << (M + max(0, S))
static inline uint32_t swizzle_apply(int B, int M, int S, uint32_t off) {
  if (B == 0) return off;
  int shiftDown = S;
  int yyyShift = M + (S > 0 ? S : 0);
  uint32_t bitMask = (1u << B) - 1;
  uint32_t yyy = bitMask << yyyShift;
  return off ^ ((off & yyy) >> shiftDown);
}

static void dump_swizzle(const char* name, int B, int M, int S) {
  const int offs[] = {0x000, 0x040, 0x080, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380, 0x400};
  const int n = sizeof(offs) / sizeof(offs[0]);
  printf("  \"%s\": {\n", name);
  for (int i = 0; i < n; ++i) {
    uint32_t phys = swizzle_apply(B, M, S, (uint32_t)offs[i]);
    printf("    \"0x%03x\": \"0x%03x\"%s\n", offs[i], phys, i + 1 < n ? "," : "");
  }
  printf("  }");
}

// ---- cute::Layout evaluator (hierarchical IntTuple) -----------------------
// We flatten shape & stride, decompose the linear idx into col-major coords
// (innermost dim cycles fastest), then compute offset = sum(coord_i * stride_i).
static uint32_t layout_at(const std::vector<int>& shape, const std::vector<int>& stride, uint32_t idx) {
  uint32_t off = 0;
  uint32_t rem = idx;
  for (size_t i = 0; i < shape.size(); ++i) {
    uint32_t c = rem % (uint32_t)shape[i];
    rem /= (uint32_t)shape[i];
    off += c * (uint32_t)stride[i];
  }
  return off;
}

struct CLayoutTest {
  const char* name;
  int M;
  int threads;
  int values_per_thread;
  std::vector<int> shape;  // flat
  std::vector<int> stride; // flat
};

static void dump_clayout(const CLayoutTest& c) {
  printf("  \"%s\": {\n", c.name);
  printf("    \"M\": %d, \"threads\": %d, \"values_per_thread\": %d,\n",
         c.M, c.threads, c.values_per_thread);
  printf("    \"thread0\": [");
  for (int v = 0; v < c.values_per_thread; ++v) {
    uint32_t idx = 0 + v * c.threads;
    uint32_t off = layout_at(c.shape, c.stride, idx);
    int m = off % c.M;
    int n = off / c.M;
    printf("[%d,%d]%s", m, n, v + 1 < c.values_per_thread ? "," : "");
  }
  printf("]\n  }");
}

int main() {
  // Byte-level Swizzle params = upcast<8> of the cute bit-level atom.
  // cute::Swizzle<3,4,3> (bit-level, 128B span) → Swizzle<3,1,3> (byte-level).
  printf("{\n\"swizzle\": {\n");
  dump_swizzle("NO_SWIZZLE",    0, 1, 3); printf(",\n");
  dump_swizzle("SW32",          1, 1, 3); printf(",\n");
  dump_swizzle("SW64",          2, 1, 3); printf(",\n");
  dump_swizzle("SW128",         3, 1, 3); printf(",\n");
  dump_swizzle("SW128_BASE32B", 2, 2, 2); printf("\n");
  printf("},\n\"clayout\": {\n");

  // SM80_16x8_Row: Layout<Shape<Shape<_4,_8>,Shape<_2,_2>>, Stride<Stride<_32,_1>,Stride<_16,_8>>>
  dump_clayout({"SM80_16x8_Row", 16, 32, 4, {4, 8, 2, 2}, {32, 1, 16, 8}});
  printf(",\n");

  // CLayout_64xN<N>: Shape<Shape<_4,_8,_4>,Shape<_2,_2,N/8>>, Stride<Stride<_128,_1,_16>,Stride<_64,_8,_512>>
  dump_clayout({"CLayout_64x8", 64, 128, 4, {4, 8, 4, 2, 2, 1}, {128, 1, 16, 64, 8, 512}});
  printf(",\n");
  dump_clayout({"CLayout_64x128", 64, 128, 64, {4, 8, 4, 2, 2, 16}, {128, 1, 16, 64, 8, 512}});
  printf("\n}\n}\n");
  return 0;
}
