#pragma once

#include <immintrin.h>

//===----------------------------------------------------------------------===//
inline __mmask8
cnt_to_mask8(const uint32_t cnt) {
  return __mmask8((1u << cnt) - 1);
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
union alignas(64)
r512 {
  uint64_t u64[8];
  int64_t  i64[8];

  uint32_t u32[16];
  int32_t  i32[16];

  uint16_t u16[32];
  int16_t  i16[32];

  uint8_t   u8[64];
  int8_t    i8[64];

  __m512i i; // integer data
  __m512  s; // single precision floating point data
  __m512d d; // double precision floating point data
};
//===----------------------------------------------------------------------===//
