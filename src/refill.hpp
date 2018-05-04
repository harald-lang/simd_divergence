#pragma once

#include <cassert>
#include <cstdint>
#include <immintrin.h>


//===----------------------------------------------------------------------===//
// Constants
static const __m512i ZERO = _mm512_setzero_si512();
static const __m512i ALL = _mm512_set1_epi64(-1);
static const __m512i SEQUENCE = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
static const uint32_t LANE_CNT = 8;
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Refill algorithm that is generally applicable. Neither the source nor the
/// destination registers need to be in a compressed state.
struct fill_random_random {
  __mmask8 permutation_mask;
  __m512i permutation_idxs;

  fill_random_random(const __mmask8 src_mask, const __mmask8 dst_mask) {
    // prepare the permutation
    const __m512i src_idxs = _mm512_mask_compress_epi64(ALL, src_mask, SEQUENCE);
    const __mmask8 write_mask = __mmask8(_mm512_knot(dst_mask));
    permutation_idxs = _mm512_mask_expand_epi64(ALL, write_mask, src_idxs);
    permutation_mask = _mm512_mask_cmpneq_epu64_mask(write_mask, permutation_idxs, ALL);
  }

  /// Update destination mask.
  inline void
  update_dst_mask(__mmask8& dst_mask) const {
    dst_mask = __mmask8(_mm512_kor(dst_mask, permutation_mask));
  }

  /// Update source mask.
  inline void
  update_src_mask(__mmask8& src_mask) const {
    const __mmask8 compressed_mask = __mmask8(_pext_u32(~0u, permutation_mask));
    const __m512i a = _mm512_maskz_mov_epi64(compressed_mask, ALL);
    const __m512i b = _mm512_maskz_expand_epi64(src_mask, a);
    src_mask = _mm512_mask_cmpeq_epu64_mask(src_mask, b, ZERO);
  }

  /// Move elements from 'src' to 'dst'.
  inline void
  operator()(const __m512i src, __m512i& dst) const {
    dst = _mm512_mask_permutexvar_epi64(dst, permutation_mask, permutation_idxs, src);
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Refill algorithm applicable where both the source and destination register
/// are compressed.
struct fill_compressed_compressed {
  __mmask8 permutation_mask;
  __m512i permutation_idxs;
  uint32_t cnt;

  /// Prepare the permutation.
  fill_compressed_compressed(const uint32_t src_cnt, const uint32_t dst_cnt) {
    const auto src_empty_cnt = LANE_CNT - src_cnt;
    const auto dst_empty_cnt = LANE_CNT - dst_cnt;

    // Determine the number of elements to move.
    cnt = std::min(src_cnt, dst_empty_cnt);
    auto d = (dst_empty_cnt >= src_cnt) ? dst_cnt : src_empty_cnt;
    const __m512i d_vec = _mm512_set1_epi64(d);

    permutation_idxs = _mm512_sub_epi64(SEQUENCE, d_vec);
    permutation_mask = ((1u << cnt) - 1) << dst_cnt;
  }

  /// Update destination count.
  inline void
  update_dst_cnt(uint32_t& dst_cnt) const {
    dst_cnt += cnt;
  }

  /// Update source count.
  inline void
  update_src_cnt(uint32_t& src_cnt) const {
    src_cnt -= cnt;
  }

  /// Move elements from 'src' to 'dst'.
  inline void
  operator()(const __m512i src, __m512i& dst) const {
    dst = _mm512_mask_permutexvar_epi64(dst, permutation_mask, permutation_idxs, src);
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Refill algorithm applicable where both the source and destination register
/// are compressed.
/// All active elements in src need to fit in dst.
struct fill_compressed_compressed_all {
  __mmask8 permutation_mask;
  __m512i permutation_idxs;
  uint32_t cnt;

  /// Prepare the permutation.
  fill_compressed_compressed_all(const uint32_t src_cnt, const uint32_t dst_cnt) {
    assert((src_cnt + dst_cnt) <= LANE_CNT);
    cnt = src_cnt;
    const __m512i d_vec = _mm512_set1_epi64(dst_cnt);
    permutation_idxs = _mm512_sub_epi64(SEQUENCE, d_vec);
    permutation_mask = ((1u << cnt) - 1) << dst_cnt;
  }

  /// Update destination count.
  inline void
  update_dst_cnt(uint32_t& dst_cnt) const {
    dst_cnt += cnt;
  }

  /// Update source count.
  inline void
  update_src_cnt(uint32_t& src_cnt) const {
    src_cnt = 0u;
  }

  /// Move elements from 'src' to 'dst'.
  inline void
  operator()(const __m512i src, __m512i& dst) const {
    dst = _mm512_mask_permutexvar_epi64(dst, permutation_mask, permutation_idxs, src);
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Refill algorithm applicable where the source register is compressed and
/// all elements from src fit into dst.
struct fill_random_compressed_all {
  __mmask8 permutation_mask;
  __m512i permutation_idxs;
  uint32_t cnt;

  /// Prepare the permutation.
  fill_random_compressed_all(const __mmask8 src_mask, const uint32_t dst_cnt) {
    assert(__builtin_popcount(src_mask) <= (LANE_CNT - dst_cnt));
    // prepare the permutation
    const __m512i src_idxs = _mm512_mask_compress_epi64(ALL, src_mask, SEQUENCE);
    const __mmask8 write_mask = __mmask8(-1u) << dst_cnt;
    permutation_idxs = _mm512_mask_expand_epi64(ALL, write_mask, src_idxs);
    permutation_mask = _mm512_mask_cmpneq_epu64_mask(write_mask, permutation_idxs, ALL);
    cnt = __builtin_popcount(src_mask);
  }

  /// Update source mask.
  inline void
  update_src_mask(__mmask8& src_mask) const {
    src_mask = __mmask8(0);
  }

  /// Update destination count.
  inline void
  update_dst_cnt(uint32_t& dst_cnt) const {
    dst_cnt += cnt;
  }

  /// Move elements from 'src' to 'dst'.
  inline void
  operator()(const __m512i src, __m512i& dst) const {
    dst = _mm512_mask_permutexvar_epi64(dst, permutation_mask, permutation_idxs, src);
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Refill algorithm applicable where the src register is compressed.
struct fill_compressed_random {
  __mmask8 permutation_mask;
  __m512i permutation_idxs;

  /// Prepare the permutation.
  fill_compressed_random(uint32_t& src_cnt, const __mmask8 dst_mask) {
    const auto dst_cnt = __builtin_popcount(dst_mask);
    const uint32_t dst_empty_cnt = LANE_CNT - dst_cnt;
    const auto src_remain_cnt = (src_cnt > dst_empty_cnt) ? src_cnt - dst_empty_cnt : 0;
    const __mmask8 src_mask = __mmask8(((1u << (src_cnt - src_remain_cnt)) - 1) << src_remain_cnt);
    const __m512i src_idxs = _mm512_mask_compress_epi64(ALL, src_mask, SEQUENCE);
    const __mmask8 write_mask = __mmask8(_mm512_knot(dst_mask));
    permutation_idxs = _mm512_mask_expand_epi64(ALL, write_mask, src_idxs);
    permutation_mask = _mm512_mask_cmpneq_epu64_mask(write_mask, permutation_idxs, ALL);
    src_cnt = src_remain_cnt;
  }

  /// Update destination mask.
  inline void
  update_dst_mask(__mmask8& dst_mask) const {
    dst_mask = __mmask8(_mm512_kor(dst_mask, permutation_mask));
  }

  /// Move elements from 'src' to 'dst'.
  inline void
  operator()(const __m512i src, __m512i& dst) const {
    dst = _mm512_mask_permutexvar_epi64(dst, permutation_mask, permutation_idxs, src);
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// Refill algorithm applicable where the src register is compressed.
/// All elements from src fit into dst.
struct fill_compressed_random_all {
  __mmask8 permutation_mask;
  __m512i permutation_idxs;

  /// Prepare the permutation.
  fill_compressed_random_all(const uint32_t src_cnt, const __mmask8 dst_mask) {
    const __mmask8 src_mask = __mmask8(((1u << src_cnt) - 1));
    const __m512i src_idxs = _mm512_mask_mov_epi64(ALL, src_mask, SEQUENCE);
    const __mmask8 write_mask = __mmask8(_mm512_knot(dst_mask));
    permutation_idxs = _mm512_mask_expand_epi64(ALL, write_mask, src_idxs);
    permutation_mask = _mm512_mask_cmpneq_epu64_mask(write_mask, permutation_idxs, ALL);
  }

  /// Update source count.
  inline void
  update_src_cnt(uint32_t& src_cnt) const {
    src_cnt = 0;
  }

  /// Update destination mask.
  inline void
  update_dst_mask(__mmask8& dst_mask) const {
    dst_mask = __mmask8(_mm512_kor(dst_mask, permutation_mask));
  }

  /// Move elements from 'src' to 'dst'.
  inline void
  operator()(const __m512i src, __m512i& dst) const {
    dst = _mm512_mask_permutexvar_epi64(dst, permutation_mask, permutation_idxs, src);
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
using fill_rr = fill_random_random;
using fill_rc_all = fill_random_compressed_all;
using fill_cr = fill_compressed_random;
using fill_cr_all = fill_compressed_random_all;
using fill_cc = fill_compressed_compressed;
using fill_cc_all = fill_compressed_compressed_all;
//===----------------------------------------------------------------------===//
