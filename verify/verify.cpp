// verify.cpp — Tier 3 golden-values harness.
//
// Compile against real cute headers and print JSON lines that the TS port
// must reproduce exactly. If the TS implementation diverges, diff the JSON
// against `expected.json` in this directory.
//
// Build (no CUDA toolkit required — cute/swizzle and cute/layout are
// host-compilable):
//
//   g++ -std=c++17 -I/tmp/cutlass/include -O0 verify.cpp -o verify
//   ./verify > actual.json
//   diff expected.json actual.json
//
// If your cutlass checkout is elsewhere, change the -I path.

#include <cute/swizzle.hpp>
#include <cute/layout.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cstdio>
#include <cstdint>

using cute::Swizzle;
using cute::_1;
using cute::_2;
using cute::_4;
using cute::_8;
using cute::_16;
using cute::_32;
using cute::_64;
using cute::_128;
using cute::_512;

template <int B, int M, int S>
static void dump_swizzle(const char* name) {
  Swizzle<B, M, S> sw;
  printf("  \"%s\": {\n", name);
  const int offs[] = {0x000, 0x040, 0x080, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380, 0x400};
  const int n = sizeof(offs) / sizeof(offs[0]);
  for (int i = 0; i < n; ++i) {
    int off = offs[i];
    int phys = (int)sw(off);
    printf("    \"0x%03x\": \"0x%03x\"%s\n", off, phys, i + 1 < n ? "," : "");
  }
  printf("  }%s\n", "");
}

template <class Layout>
static void dump_clayout_t0(const char* name, int M, int threads, int values) {
  Layout layout;
  printf("  \"%s\": {\n", name);
  printf("    \"M\": %d, \"threads\": %d, \"values_per_thread\": %d,\n", M, threads, values);
  printf("    \"thread0\": [");
  for (int v = 0; v < values; ++v) {
    int off = (int)layout(0 + v * threads);
    int m = off % M;
    int n = off / M;
    printf("[%d,%d]%s", m, n, v + 1 < values ? "," : "");
  }
  printf("]\n  }%s\n", "");
}

int main() {
  printf("{\n");

  // --- 1. Swizzle apply() golden values.
  printf("\"swizzle\": {\n");
  dump_swizzle<0, 4, 3>("NO_SWIZZLE");
  printf(",\n");
  dump_swizzle<1, 4, 3>("SW32");
  printf(",\n");
  dump_swizzle<2, 4, 3>("SW64");
  printf(",\n");
  dump_swizzle<3, 4, 3>("SW128");
  printf(",\n");
  dump_swizzle<2, 5, 2>("SW128_BASE32B");
  printf("},\n");

  // --- 2. CLayout thread-0 ownership.
  printf("\"clayout\": {\n");
  using SM80_16x8_Row = cute::SM80_16x8_Row;
  dump_clayout_t0<SM80_16x8_Row>("SM80_16x8_Row", 16, 32, 4);
  printf(",\n");
  using CL64x8 = cute::SM90::GMMA::CLayout_64x8;
  dump_clayout_t0<CL64x8>("CLayout_64x8", 64, 128, 4);
  printf(",\n");
  using CL64x128 = cute::SM90::GMMA::CLayout_64x128;
  dump_clayout_t0<CL64x128>("CLayout_64x128", 64, 128, 64);
  printf("}\n");

  printf("}\n");
  return 0;
}
