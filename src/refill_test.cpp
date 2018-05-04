#ifndef __AVX512F__
#error AVX-512 not supported.
#endif

#include <cassert>
#include <fstream>
#include <iostream>

#include "refill.hpp"
#include "util.hpp"


//===----------------------------------------------------------------------===//
// Scalar refill implementations (for testing purposes).
namespace scalar {

void
fill_cc(const r512& src, uint32_t& src_cnt, r512& dst, uint32_t& dst_cnt) {
  const auto move_cnt = std::min(LANE_CNT - dst_cnt, src_cnt);
  for (uint32_t i = 0; i < move_cnt; i++) {
    uint32_t wp = dst_cnt + i;
    uint32_t rp = src_cnt - move_cnt + i;
    dst.u64[wp] = src.u64[rp];
  }
  src_cnt -= move_cnt;
  dst_cnt += move_cnt;
}
void
fill_cc(const __m512i& src, uint32_t& src_cnt, __m512i& dst, uint32_t& dst_cnt) {
  fill_cc(reinterpret_cast<const r512&>(src), src_cnt, reinterpret_cast<r512&>(dst), dst_cnt);
}

void
fill_rc(const r512& src, __mmask8& src_mask, r512& dst, uint32_t& dst_cnt) {
  uint32_t move_cntr = 0;
  uint32_t wp = dst_cnt;
  while (src_mask != 0 && wp < LANE_CNT) {
    const uint32_t rp = __builtin_ctz(src_mask);
    dst.u64[wp] = src.u64[rp];
    move_cntr++;
    wp++;
    src_mask = src_mask & (src_mask - __mmask8(1));
  }
  dst_cnt += move_cntr;
}
void
fill_rc(const __m512i& src, __mmask8& src_mask, __m512i& dst, uint32_t& dst_cnt) {
  fill_rc(reinterpret_cast<const r512&>(src), src_mask, reinterpret_cast<r512&>(dst), dst_cnt);
}

void
fill_rr(const r512& src, __mmask8& src_mask, r512& dst, __mmask8& dst_mask) {
  const uint32_t move_cnt = std::min(0u + __builtin_popcount(src_mask), LANE_CNT - __builtin_popcount(dst_mask));
  __mmask8 dst_write_mask = ~dst_mask;
  for (uint32_t i = 0; i < move_cnt; i++) {
    const uint32_t rp = __builtin_ctz(src_mask);
    const uint32_t wp = __builtin_ctz(dst_write_mask);
    dst.u64[wp] = src.u64[rp];
    src_mask = src_mask & (src_mask - __mmask8(1));
    dst_write_mask = dst_write_mask & (dst_write_mask - __mmask8(1));
  }
  dst_mask |= ~dst_write_mask;
}
void
fill_rr(const __m512i& src, __mmask8& src_mask, __m512i& dst, __mmask8& dst_mask) {
  fill_rr(reinterpret_cast<const r512&>(src), src_mask, reinterpret_cast<r512&>(dst), dst_mask);
}

void
fill_cr(const r512& src, uint32_t& src_cnt, r512& dst, __mmask8& dst_mask) {
  const uint32_t move_cnt = std::min(src_cnt, LANE_CNT - __builtin_popcount(dst_mask));
  __mmask8 dst_write_mask = ~dst_mask;
  for (uint32_t i = 0; i < move_cnt; i++) {
    const uint32_t rp = src_cnt - move_cnt + i;
    const uint32_t wp = __builtin_ctz(dst_write_mask);
    dst.u64[wp] = src.u64[rp];
    dst_write_mask = dst_write_mask & (dst_write_mask - __mmask8(1));
  }
  src_cnt -= move_cnt;
  dst_mask |= ~dst_write_mask;
}
void
fill_cr(const __m512i& src, uint32_t& src_cnt, __m512i& dst, __mmask8& dst_mask) {
  fill_cr(reinterpret_cast<const r512&>(src), src_cnt, reinterpret_cast<r512&>(dst), dst_mask);
}

} // namespace scalar
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
test_cc(uint32_t in_src_cnt, uint32_t in_dst_cnt) {
  __m512i src = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i dst = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

  uint32_t src_cnt = in_src_cnt;
  uint32_t dst_cnt = in_dst_cnt;

  __m512i expected_dst = dst;
  uint32_t expected_src_cnt = in_src_cnt;
  uint32_t expected_dst_cnt = in_dst_cnt;
  scalar::fill_cc(src, expected_src_cnt, expected_dst, expected_dst_cnt);

  fill_cc fill(src_cnt, dst_cnt);
  fill(src, dst);
  fill.update_src_cnt(src_cnt);
  fill.update_dst_cnt(dst_cnt);

  __mmask8 res = _mm512_cmpeq_epi64_mask(expected_dst, dst);
  assert(res == cnt_to_mask8(LANE_CNT));
  assert(src_cnt == expected_src_cnt);
  assert(dst_cnt == expected_dst_cnt);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
test_cc_all(uint32_t src_cnt, uint32_t dst_cnt) {
  assert((src_cnt + dst_cnt) <= LANE_CNT);
  __m512i src = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i dst = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);;

  __m512i expected_dst = dst;
  uint32_t expected_src_cnt = src_cnt;
  uint32_t expected_dst_cnt = dst_cnt;
  scalar::fill_cc(src, expected_src_cnt, expected_dst, expected_dst_cnt);

  fill_cc_all fill(src_cnt, dst_cnt);
  fill(src, dst);
  fill.update_src_cnt(src_cnt);
  fill.update_dst_cnt(dst_cnt);

  __mmask8 res = _mm512_cmpeq_epi64_mask(expected_dst, dst);
  assert(res == cnt_to_mask8(LANE_CNT));
  assert(src_cnt == expected_src_cnt);
  assert(dst_cnt == expected_dst_cnt);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
test_rc_all(__mmask8 src_mask, uint32_t dst_cnt) {
  assert((__builtin_popcount(src_mask) + dst_cnt) <= LANE_CNT);
  __m512i src = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i dst = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);;

  __m512i expected_dst = dst;
  __mmask8 expected_src_mask = src_mask;
  uint32_t expected_dst_cnt = dst_cnt;
  scalar::fill_rc(src, expected_src_mask, expected_dst, expected_dst_cnt);

  fill_rc_all fill(src_mask, dst_cnt);
  fill(src, dst);
  fill.update_src_mask(src_mask);
  fill.update_dst_cnt(dst_cnt);

  __mmask8 res = _mm512_cmpeq_epi64_mask(expected_dst, dst);
  assert(res == cnt_to_mask8(LANE_CNT));
  assert(src_mask == expected_src_mask);
  assert(dst_cnt == expected_dst_cnt);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
test_rr(__mmask8 src_mask, __mmask8 dst_mask) {
  __m512i src = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i dst = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);;

  __m512i expected_dst = dst;
  __mmask8 expected_src_mask = src_mask;
  __mmask8 expected_dst_mask = dst_mask;
  scalar::fill_rr(src, expected_src_mask, expected_dst, expected_dst_mask);

  fill_rr fill(src_mask, dst_mask);
  fill(src, dst);
  fill.update_src_mask(src_mask);
  fill.update_dst_mask(dst_mask);

  __mmask8 res = _mm512_cmpeq_epi64_mask(expected_dst, dst);
  assert(res == cnt_to_mask8(LANE_CNT));
  assert(src_mask == expected_src_mask);
  assert(dst_mask == expected_dst_mask);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
test_cr(uint32_t src_cnt, __mmask8 dst_mask) {
  __m512i src = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i dst = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);;

  __m512i expected_dst = dst;
  uint32_t expected_src_cnt = src_cnt;
  __mmask8 expected_dst_mask = dst_mask;
  scalar::fill_cr(src, expected_src_cnt, expected_dst, expected_dst_mask);

  fill_cr fill(src_cnt, dst_mask);
  fill(src, dst);
  fill.update_dst_mask(dst_mask);

  __mmask8 res = _mm512_cmpeq_epi64_mask(expected_dst, dst);
  assert(res == cnt_to_mask8(LANE_CNT));
  assert(src_cnt == expected_src_cnt);
  assert(dst_mask == expected_dst_mask);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
void
test_cr_all(uint32_t src_cnt, __mmask8 dst_mask) {
  assert((src_cnt + __builtin_popcount(dst_mask)) <= LANE_CNT);
  __m512i src = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i dst = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);;

  __m512i expected_dst = dst;
  uint32_t expected_src_cnt = src_cnt;
  __mmask8 expected_dst_mask = dst_mask;
  scalar::fill_cr(src, expected_src_cnt, expected_dst, expected_dst_mask);

  fill_cr_all fill(src_cnt, dst_mask);
  fill(src, dst);
  fill.update_src_cnt(src_cnt);
  fill.update_dst_mask(dst_mask);

  __mmask8 res = _mm512_cmpeq_epi64_mask(expected_dst, dst);
  assert(res == cnt_to_mask8(LANE_CNT));
  assert(src_cnt == expected_src_cnt);
  assert(dst_mask == expected_dst_mask);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
int32_t
main() {

  {
    std::size_t cntr = 0;
    std::cout << "Testing: random -> random (generic): ";
    for (uint32_t src_mask = 0; src_mask < cnt_to_mask8(8); src_mask++) {
      for (uint32_t dst_mask = 0; dst_mask < cnt_to_mask8(8); dst_mask++) {
        test_rr(src_mask, dst_mask);
        cntr++;
      }
    }
    std::cout << cntr << " tests passed." << std::endl;
  }

  {
    std::size_t cntr = 0;
    std::cout << "Testing: random -> compressed (all fit): ";
    for (uint32_t src_mask = 0; src_mask < cnt_to_mask8(8); src_mask++) {
      for (uint32_t dst_cnt = 0; dst_cnt < 8; dst_cnt++) {
        if ((__builtin_popcount(src_mask) + dst_cnt) > LANE_CNT) continue;
        test_rc_all(src_mask, dst_cnt);
        cntr++;
      }
    }
    std::cout << cntr << " tests passed." << std::endl;
  }

  {
    std::size_t cntr = 0;
    std::cout << "Testing: compressed -> compressed (generic): ";
    for (uint32_t src_cnt = 0; src_cnt < 8; src_cnt++) {
      for (uint32_t dst_cnt = 0; dst_cnt < 8; dst_cnt++) {
        test_cc(src_cnt, dst_cnt);
        cntr++;
      }
    }
    std::cout << cntr << " tests passed." << std::endl;
  }


  {
    std::size_t cntr = 0;
    std::cout << "Testing: compressed -> compressed (all fit): ";
    for (uint32_t src_cnt = 0; src_cnt < 8; src_cnt++) {
      for (uint32_t dst_cnt = 0; dst_cnt < 8; dst_cnt++) {
        if ((src_cnt + dst_cnt) > LANE_CNT) continue;
        test_cc_all(src_cnt, dst_cnt);
        cntr++;
      }
    }
    std::cout << cntr << " tests passed." << std::endl;
  }

  {
    std::size_t cntr = 0;
    std::cout << "Testing: compressed -> random (generic): ";
    for (uint32_t src_cnt = 0; src_cnt < 8; src_cnt++) {
      for (uint32_t dst_mask = 0; dst_mask < cnt_to_mask8(8); dst_mask++) {
        test_cr(src_cnt, dst_mask);
        cntr++;
      }
    }
    std::cout << cntr << " tests passed." << std::endl;
  }

  {
    std::size_t cntr = 0;
    std::cout << "Testing: compressed -> random (all): ";
    for (uint32_t src_cnt = 0; src_cnt < 8; src_cnt++) {
      for (uint32_t dst_mask = 0; dst_mask < cnt_to_mask8(8); dst_mask++) {
        if(src_cnt + __builtin_popcount(dst_mask) > LANE_CNT) continue;
        test_cr_all(src_cnt, dst_mask);
        cntr++;
      }
    }
    std::cout << cntr << " tests passed." << std::endl;
  }

  std::cout << "Done." << std::endl;
  return 0;
}
//===----------------------------------------------------------------------===//
