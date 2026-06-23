// ======================================================================================
// * Copyright (c) 2026, D.Skryabin / tg @ai_bond007 SPDX-License: BSD-3-Clause
// ======================================================================================
#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include "kernel.h"

#ifndef KERNEL_DEBUG
    #define KERNEL_DEBUG 0
#endif

#if KERNEL_DEBUG
    #define __ASM_MARK(STG, CTX, TYPE, MAGIC) \
        do { \
            volatile unsigned int point = 0; \
            asm volatile("mov.u32 %0, " #MAGIC "; // DBG_PTX_" #STG "_" #CTX "_" #TYPE "\n\t" \
                         : "+r"(point) :: "memory"); \
            (void)point; \
        } while(0)

    #define __ASM_DEBUG_BEGIN(STG, CTX) __ASM_MARK(STG, CTX, BEGIN, 0xBEEF0001)
    #define __ASM_DEBUG_END(STG, CTX)   __ASM_MARK(STG, CTX, END,   0xCAFE0002)

enum DebugType : uint8_t {
    SMEM     = 0, TILE     = 1, SQKT     = 2, DOVT     = 3, DOPV     = 4,
    DQDSK    = 5, DVPTDO   = 6, DKDSTQ   = 7, SOFTMAX  = 8, ROWDQ    = 9,
    ROWDKV   = 10, WRITEO  = 11, WRITEQ  = 12, WRITEKV = 13, NONE    = 14
};

__device__ __forceinline__ const char* stage_name(DebugType stage) {
    switch(stage) {
        case SMEM:    return "SMEM";    case TILE:    return "TILE";
        case SQKT:    return "SQKT";    case DOVT:    return "DOVT";
        case DOPV:    return "DOPV";    case DQDSK:   return "DQDSK";
        case DVPTDO:  return "DVPTDO";  case DKDSTQ:  return "DKDSTQ";
        case SOFTMAX: return "SOFTMAX"; case ROWDQ:   return "ROWDQ";
        case ROWDKV:  return "ROWDKV";  case WRITEO:  return "WRITEO";
        case WRITEQ:  return "WRITEQ";  case WRITEKV: return "WRITEKV";
        case NONE:    return "NONE";    default:      return "????";
    }
}

template<typename T> struct element_type { using type = T; };
template<typename T> struct element_type<T[]> { using type = T; };
template<typename T, size_t N> struct element_type<T[N]> { using type = T; };
template<typename T> using element_type_t = typename element_type<T>::type;

template<typename Layout, typename FieldTag, typename = void>
struct field_info {};

#define DEFINE_FIELD(FIELD_NAME, FIELD_PATH) \
    template<typename T, typename = void> \
    struct has_##FIELD_NAME : std::false_type {}; \
    template<typename T> \
    struct has_##FIELD_NAME<T, std::void_t<decltype(std::declval<T>().FIELD_PATH)>> : std::true_type {}; \
    struct TAG_##FIELD_NAME {}; \
    template<typename Layout> \
    struct field_info<Layout, TAG_##FIELD_NAME> { \
        using type = element_type_t<decltype(std::declval<Layout>().FIELD_PATH)>; \
        __device__ static constexpr size_t get_offset() { \
            return offsetof(Layout, FIELD_PATH); \
        } \
    };

// Forward fields
DEFINE_FIELD(q_fwd,       phase.fdo.q)
DEFINE_FIELD(k_fwd,       phase.fdo.reuse_kv.k)
DEFINE_FIELD(v_fwd,       phase.fdo.reuse_kv.v)
DEFINE_FIELD(s_fwd,       phase.fdo.reuse_sp.s)
DEFINE_FIELD(p_fwd,       phase.fdo.reuse_sp.p)
DEFINE_FIELD(row_max_fwd, row_max)
DEFINE_FIELD(row_sum_fwd, row_sum)
DEFINE_FIELD(o_fwd,       phase.fdo.o)

// Backward dQ fields
DEFINE_FIELD(q_dq,        phase.bdq.q)
DEFINE_FIELD(k_dq,        phase.bdq.reuse_kv.k)
DEFINE_FIELD(v_dq,        phase.bdq.reuse_kv.v)
DEFINE_FIELD(s_dq,        phase.bdq.s)
DEFINE_FIELD(dO_dq,       phase.bdq.dO)
DEFINE_FIELD(dOV_dq,      phase.bdq.reuse_sdOVS.dOV)
DEFINE_FIELD(dS_dq,       phase.bdq.reuse_sdOVS.dS)
DEFINE_FIELD(dQ_dq,       phase.bdq.dQ)

// Backward dKV fields
DEFINE_FIELD(k_dkv,       phase.bdkv.k)
DEFINE_FIELD(v_dkv,       phase.bdkv.v)
DEFINE_FIELD(q_dkv,       phase.bdkv.reuse_qdO.q)
DEFINE_FIELD(dO_dkv,      phase.bdkv.reuse_qdO.dO)
DEFINE_FIELD(s_dkv,       phase.bdkv.reuse_sp.s)
DEFINE_FIELD(p_dkv,       phase.bdkv.reuse_sp.p)
DEFINE_FIELD(dS_dkv,      phase.bdkv.reuse_dOVS.dS)
DEFINE_FIELD(dOV_dkv,     phase.bdkv.reuse_dOVS.dOV)
DEFINE_FIELD(dK_dkv,      phase.bdkv.dK)
DEFINE_FIELD(dV_dkv,      phase.bdkv.dV)

// Common vectors
DEFINE_FIELD(lse,         lse)
DEFINE_FIELD(row_dot,     row_dot)
#undef DEFINE_FIELD

__device__ __forceinline__ void _print_val(float v) {
    if (isnan(v))             printf("    nan ");
    else if (isinf(v) || v <= -1e20f) printf("   -inf ");
    else                      printf("%7.3f ", v);
}

// ======================================================================================
// __CHECK_INIT: Validate SMEM zero-fill
// ======================================================================================
#define __CHECK_INIT(FIELD_TAG, EXPECTED_VAL, VALID_ROWS) \
do { \
    extern __shared__ char smem_raw[]; \
    uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); \
    using Layout = typename Config::SmemLayout; \
    constexpr bool has_field = has_##FIELD_TAG<Layout>::value; \
    if constexpr (has_field) { \
        using Info = field_info<Layout, TAG_##FIELD_TAG>; \
        using FieldType = typename Info::type; \
        bool mismatch = false; \
        if (tid < (VALID_ROWS)) { \
            size_t offset = Info::get_offset(); \
            uint32_t elem_addr = base + static_cast<uint32_t>(offset) + (tid * sizeof(FieldType)); \
            if constexpr (std::is_same<FieldType, __half>::value) { \
                uint16_t val_bits, exp_bits; \
                asm volatile("ld.shared.u16 %0, [%1];" : "=h"(val_bits) : "r"(elem_addr)); \
                union { __half h; uint16_t u; } conv; conv.h = __float2half_rn(EXPECTED_VAL); exp_bits = conv.u; \
                mismatch = (val_bits != exp_bits); \
            } else if constexpr (std::is_same<FieldType, float>::value) { \
                uint32_t val_bits, exp_bits; \
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(val_bits) : "r"(elem_addr)); \
                union { float f; uint32_t u; } conv; conv.f = EXPECTED_VAL; exp_bits = conv.u; \
                mismatch = (val_bits != exp_bits); \
            } \
            if (mismatch && (tid & 31) == 0) { \
                printf("[DBG_ERR][B%d][INIT]: " #FIELD_TAG "[%d] MISMATCH @ 0x%x\n", blockIdx.x, tid, elem_addr); \
            } \
        } \
    } \
} while(0)

// ======================================================================================
// __CHECK_ERRORS: Stage-aware inf/nan scan
// ======================================================================================
#define __CHECK_ERRORS(STG, VM, VN, SC, WID, LID, TID) \
do { \
    if ((TID) == 0 && (blockIdx.x == 0 && blockIdx.z == 0)) { \
        extern __shared__ char smem_raw[]; \
        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw); \
        using Layout = typename Config::SmemLayout; \
        bool found_err = false; \
        int rows = (VM) < WMMA_M ? (VM) : WMMA_M; \
        int cols = (VN) < WMMA_N ? (VN) : WMMA_N; \
        for (int r = 0; r < rows; ++r) { \
            for (int c = 0; c < cols; ++c) { \
                float val = 0.f; \
                switch(STG) { \
                    case SQKT: case DOVT: \
                        if constexpr (has_s_fwd<Layout>::value) val = smem.phase.fdo.reuse_sp.s[r * (SC) + c]; \
                        else if constexpr (has_s_dq<Layout>::value) val = smem.phase.bdq.s[r * (SC) + c]; \
                        else if constexpr (has_s_dkv<Layout>::value) val = smem.phase.bdkv.reuse_sp.s[r * (SC) + c]; \
                        break; \
                    case SOFTMAX: \
                        if constexpr (has_p_fwd<Layout>::value) val = __half2float(smem.phase.fdo.reuse_sp.p[r * (SC) + c]); \
                        else if constexpr (has_p_dkv<Layout>::value) val = __half2float(smem.phase.bdkv.reuse_sp.p[r * (SC) + c]); \
                        break; \
                    case DQDSK: case DVPTDO: case DKDSTQ: \
                        if constexpr (has_dS_dq<Layout>::value) val = __half2float(smem.phase.bdq.reuse_sdOVS.dS[r * (SC) + c]); \
                        else if constexpr (has_dS_dkv<Layout>::value) val = __half2float(smem.phase.bdkv.reuse_dOVS.dS[r * (SC) + c]); \
                        break; \
                    case DOPV: case WRITEO: \
                        if constexpr (has_o_fwd<Layout>::value) val = smem.phase.fdo.o[r * (SC) + c]; \
                        break; \
                    case WRITEQ: \
                        if constexpr (has_dQ_dq<Layout>::value) val = smem.phase.bdq.dQ[r * (SC) + c]; \
                        break; \
                    case WRITEKV: \
                        if constexpr (has_dK_dkv<Layout>::value) val = smem.phase.bdkv.dK[r * (SC) + c]; \
                        else if constexpr (has_dV_dkv<Layout>::value) val = smem.phase.bdkv.dV[r * (SC) + c]; \
                        break; \
                    default: break; \
                } \
                if (isnan(val) || isinf(val)) { \
                    printf("[DBG_ERR][B%d][%s][%d,%d]: inf/nan\n", blockIdx.x, stage_name(STG), r, c); \
                    found_err = true; \
                } \
            } \
        } \
        if (!found_err) printf("[DBG_OK ][B%d][%s] tile checked\n", blockIdx.x, stage_name(STG)); \
    } \
} while(0)

// ======================================================================================
// __PRINT_MATRIX: 2D tile dump
// ======================================================================================
#define __PRINT_MATRIX(STG, VM, VN, SC, WID, LID, TID, TILE_IDX) \
do { \
    if ((TID) == 0 && (WID) == 0 && (blockIdx.x == 0 && blockIdx.z == 0)) { \
        extern __shared__ char smem_raw[]; \
        auto& smem = *reinterpret_cast<typename Config::SmemLayout*>(smem_raw); \
        using Layout = typename Config::SmemLayout; \
        printf("[DBG_MAT][B%d][T%d][%s] tile[%dx%d]:\n", blockIdx.x, TILE_IDX, stage_name(STG), VM, VN); \
        int print_rows = (VM) < WMMA_M ? (VM) : WMMA_M; \
        int print_cols = (VN) < WMMA_N ? (VN) : WMMA_N; \
        for (int r = 0; r < print_rows; ++r) { \
            printf("  row %2d: ", r); \
            for (int c = 0; c < print_cols; ++c) { \
                float v = 0.f; \
                switch(STG) { \
                    case SQKT: case DOVT: \
                        if constexpr (has_s_fwd<Layout>::value) v = smem.phase.fdo.reuse_sp.s[r * (SC) + c]; \
                        else if constexpr (has_s_dq<Layout>::value) v = smem.phase.bdq.s[r * (SC) + c]; \
                        break; \
                    case DQDSK: \
                        if constexpr (has_s_dq<Layout>::value) v = smem.phase.bdq.s[r * (SC) + c]; \
                        break; \
                    case DVPTDO: \
                        if constexpr (has_dOV_dq<Layout>::value) v = __half2float(smem.phase.bdq.reuse_sdOVS.dOV[r * (SC) + c]); \
                        else if constexpr (has_dOV_dkv<Layout>::value) v = __half2float(smem.phase.bdkv.reuse_dOVS.dOV[r * (SC) + c]); \
                        break; \
                    case DKDSTQ: case DOPV: \
                        if constexpr (has_dQ_dq<Layout>::value) v = smem.phase.bdq.dQ[r * (SC) + c]; \
                        else if constexpr (has_o_fwd<Layout>::value) v = smem.phase.fdo.o[r * (SC) + c]; \
                        break; \
                    default: break; \
                } \
                _print_val(v); \
            } \
            printf("\n"); \
        } \
    } \
} while(0)

// ======================================================================================
// __PRINT_RESULT: 1D vector/scalar dump
// ======================================================================================
#define __PRINT_RESULT(FIELD_TAG, VLEN, TILE_IDX) \
do { \
    if (tid == 0 && (blockIdx.x == 0 && blockIdx.z == 0)) { \
        extern __shared__ char smem_raw[]; \
        using Layout = typename Config::SmemLayout; \
        constexpr bool has_field = has_##FIELD_TAG<Layout>::value; \
        if constexpr (has_field) { \
            using Info = field_info<Layout, TAG_##FIELD_TAG>; \
            using FieldType = typename Info::type; \
            uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw)); \
            uint32_t vec_addr = base + static_cast<uint32_t>(Info::get_offset()); \
            printf("[DBG_VEC][B%d][T%d][" #FIELD_TAG "] len=%d: ", blockIdx.x, TILE_IDX, VLEN); \
            int print_len = (VLEN) < WMMA_M ? (VLEN) : WMMA_M; \
            for (int i = 0; i < print_len; ++i) { \
                float v = 0.f; \
                if constexpr (std::is_same<FieldType, float>::value) { \
                    uint32_t bits; \
                    uint32_t elem_addr = vec_addr + (uint32_t)(i * sizeof(float)); \
                    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(bits) : "r"(elem_addr)); \
                    v = *reinterpret_cast<float*>(&bits); \
                } else if constexpr (std::is_same<FieldType, __half>::value) { \
                    uint16_t bits; \
                    uint32_t elem_addr = vec_addr + (uint32_t)(i * sizeof(__half)); \
                    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(bits) : "r"(elem_addr)); \
                    v = __half2float(*reinterpret_cast<__half*>(&bits)); \
                } \
                _print_val(v); \
            } \
            printf("\n"); \
        } \
    } \
} while(0)

// ======================================================================================
// Forward pass occupancy
// ======================================================================================
#define __PRINT_OCCUPANCY_FORWARD(D, M, N, B, H_Q, H_K, grid, block) \
do { \
    using Layout = typename Config::SmemLayout; \
    constexpr size_t smem_total = Config::TOTAL_SMEM; \
    printf("== Flash Attention Forward ===========================================\n"); \
    printf(" Tensors : D=%-3d  B=%-3d  H_Q=%-3d  H_K=%-3d  GQA=%-2d  M=%-5d  N=%-5d\n", D, B, H_Q, H_K, H_Q / H_K, M, N); \
    printf(" Kernel  : Tile=%dx%d  Threads=%d  Warps=%d  Grid=(%d,%d,%d)  Blocks=%d\n", \
           Config::DO::BLOCK_M, Config::DO::BLOCK_N, \
           Config::THREADS_PER_BLOCK, Config::WARPS_PER_BLOCK, \
           grid.x, grid.y, grid.z, grid.x * grid.y * grid.z); \
    printf(" SMEM:\n"); \
    size_t bytes_q = 0, bytes_kv = 0, bytes_sp = 0, bytes_o = 0, bytes_state = 0; \
    size_t bytes_k = 0, bytes_v = 0, bytes_s = 0, bytes_p = 0; \
    if constexpr (has_q_fwd<Layout>::value) { \
        using Info = field_info<Layout, TAG_q_fwd>; \
        bytes_q = sizeof(typename Info::type) * Config::DO::BLOCK_M * D; \
        printf("   Q           : %6zu B (%5.2f KB)\n", bytes_q, bytes_q / 1024.0f); \
    } \
    if constexpr (has_k_fwd<Layout>::value) { \
        using Info = field_info<Layout, TAG_k_fwd>; \
        bytes_k = sizeof(typename Info::type) * Config::DO::BLOCK_N * D; \
    } \
    if constexpr (has_v_fwd<Layout>::value) { \
        using Info = field_info<Layout, TAG_v_fwd>; \
        bytes_v = sizeof(typename Info::type) * Config::DO::BLOCK_N * D; \
    } \
    bytes_kv = (bytes_k > bytes_v) ? bytes_k : bytes_v; \
    if (bytes_kv > 0) { \
        printf("   K/V (reuse) : %6zu B (%5.2f KB)  [K=%zu V=%zu]\n", \
               bytes_kv, bytes_kv / 1024.0f, bytes_k, bytes_v); \
    } \
    if constexpr (has_s_fwd<Layout>::value) { \
        using Info = field_info<Layout, TAG_s_fwd>; \
        bytes_s = sizeof(typename Info::type) * Config::DO::BLOCK_M * Config::DO::BLOCK_N; \
    } \
    if constexpr (has_p_fwd<Layout>::value) { \
        using Info = field_info<Layout, TAG_p_fwd>; \
        bytes_p = sizeof(typename Info::type) * Config::DO::BLOCK_M * Config::DO::BLOCK_N; \
    } \
    bytes_sp = (bytes_s > bytes_p) ? bytes_s : bytes_p; \
    if (bytes_sp > 0) { \
        printf("   S/P (reuse) : %6zu B (%5.2f KB)  [S=%zu P=%zu]\n", \
               bytes_sp, bytes_sp / 1024.0f, bytes_s, bytes_p); \
    } \
    if constexpr (has_o_fwd<Layout>::value) { \
        using Info = field_info<Layout, TAG_o_fwd>; \
        bytes_o = sizeof(typename Info::type) * Config::DO::BLOCK_M * D; \
        printf("   O           : %6zu B (%5.2f KB)\n", bytes_o, bytes_o / 1024.0f); \
    } \
    if constexpr (has_row_max_fwd<Layout>::value && has_row_sum_fwd<Layout>::value) { \
        using InfoMax = field_info<Layout, TAG_row_max_fwd>; \
        using InfoSum = field_info<Layout, TAG_row_sum_fwd>; \
        bytes_state = sizeof(typename InfoMax::type) * Config::DO::BLOCK_M + \
                      sizeof(typename InfoSum::type) * Config::DO::BLOCK_M; \
        printf("   row_max/sum : %6zu B (%5.2f KB)\n", bytes_state, bytes_state / 1024.0f); \
    } \
    printf("   -----------------------------------------\n"); \
    printf("   TOTAL       : %6zu B (%5.2f KB)\n", smem_total, smem_total / 1024.0f); \
    const int blk_by_thr  = MAX_THREADS_PER_SM / Config::THREADS_PER_BLOCK; \
    const int blk_by_smem = (smem_total > 0) ? (MAX_SMEM_PER_SM / smem_total) : MAX_THREAD_BLOCK_PER_SM; \
    const int blk_per_sm  = min(min(blk_by_thr, blk_by_smem), MAX_THREAD_BLOCK_PER_SM); \
    const int act_warps   = blk_per_sm * Config::WARPS_PER_BLOCK; \
    const float occ       = static_cast<float>(act_warps) / MAX_WARPS_PER_SM; \
    printf(" Occupancy:\n"); \
    printf("   Blocks/SM   : %d  (thr:%d  smem:%d  hw:%d)\n", \
           blk_per_sm, blk_by_thr, blk_by_smem, MAX_THREAD_BLOCK_PER_SM); \
    printf("   Warps/SM    : %d / %d\n", act_warps, MAX_WARPS_PER_SM); \
    printf("   Occupancy   : %5.1f%%\n", occ * 100.0f); \
    printf("======================================================================\n\n"); \
} while(0)

// ======================================================================================
// Backward pass occupancy
// ======================================================================================
#define __PRINT_OCCUPANCY_BACKWARD(D, M, N, B, H_Q, H_K, grid, block, grid_dq, grid_dkv) \
do { \
    using Layout = typename Config::SmemLayout; \
    constexpr size_t smem_total = Config::TOTAL_SMEM; \
    printf("== Flash Attention Backward (fused dQ/dKV) ===========================\n"); \
    printf(" Tensors : D=%-3d  B=%-3d  H_Q=%-3d  H_K=%-3d  GQA=%-2d  M=%-5d  N=%-5d\n", D, B, H_Q, H_K, H_Q / H_K, M, N); \
    printf(" Kernel  : dQ-Tile=%dx%d  dKV-Tile=%dx%d  Threads=%d  Warps=%d\n", \
           Config::DQ::BLOCK_M,  Config::DQ::BLOCK_N, \
           Config::DKV::BLOCK_M, Config::DKV::BLOCK_N, \
           Config::THREADS_PER_BLOCK, Config::WARPS_PER_BLOCK); \
    printf(" Grid    : (%d,%d,%d)  [Y=0:dQ  Y=1:dKV]  dQ_blks=%d  dKV_blks=%d  total=%d\n", \
           grid.x, grid.y, grid.z, grid_dq, grid_dkv, grid.x * grid.y * grid.z); \
    printf(" SMEM:\n"); \
    size_t bytes_kv = 0, bytes_qdo = 0, bytes_sp = 0, bytes_dovs = 0, \
           bytes_stats = 0, bytes_dq = 0, bytes_dkv = 0; \
    size_t bytes_k = 0, bytes_v = 0, bytes_q = 0, bytes_dO = 0, \
           bytes_s = 0, bytes_p = 0, bytes_dOV = 0, bytes_dS = 0; \
    const int bm = has_q_dq<Layout>::value ? Config::DQ::BLOCK_M : Config::DKV::BLOCK_M; \
    const int bn = has_k_dq<Layout>::value ? Config::DQ::BLOCK_N : Config::DKV::BLOCK_N; \
    if constexpr (has_k_dq<Layout>::value || has_k_dkv<Layout>::value) { \
        using Info = std::conditional_t<has_k_dq<Layout>::value, field_info<Layout, TAG_k_dq>, field_info<Layout, TAG_k_dkv>>; \
        bytes_k = sizeof(typename Info::type) * bn * D; \
    } \
    if constexpr (has_v_dq<Layout>::value || has_v_dkv<Layout>::value) { \
        using Info = std::conditional_t<has_v_dq<Layout>::value, field_info<Layout, TAG_v_dq>, field_info<Layout, TAG_v_dkv>>; \
        bytes_v = sizeof(typename Info::type) * bn * D; \
    } \
    bytes_kv = (bytes_k > bytes_v) ? bytes_k : bytes_v; \
    if (bytes_kv > 0) { \
        printf("   K/V (reuse) : %6zu B (%5.2f KB)  [K=%zu V=%zu]\n", bytes_kv, bytes_kv / 1024.0f, bytes_k, bytes_v); \
    } \
    if constexpr (has_q_dq<Layout>::value || has_q_dkv<Layout>::value) { \
        using Info = std::conditional_t<has_q_dq<Layout>::value, field_info<Layout, TAG_q_dq>, field_info<Layout, TAG_q_dkv>>; \
        bytes_q = sizeof(typename Info::type) * bm * D; \
    } \
    if constexpr (has_dO_dq<Layout>::value || has_dO_dkv<Layout>::value) { \
        using Info = std::conditional_t<has_dO_dq<Layout>::value, field_info<Layout, TAG_dO_dq>, field_info<Layout, TAG_dO_dkv>>; \
        bytes_dO = sizeof(typename Info::type) * bm * D; \
    } \
    bytes_qdo = (bytes_q > bytes_dO) ? bytes_q : bytes_dO; \
    if (bytes_qdo > 0) { \
        printf("   Q/dO (reuse): %6zu B (%5.2f KB)  [Q=%zu dO=%zu]\n", bytes_qdo, bytes_qdo / 1024.0f, bytes_q, bytes_dO); \
    } \
    if constexpr (has_s_dq<Layout>::value || has_s_dkv<Layout>::value) { \
        using Info = std::conditional_t<has_s_dq<Layout>::value, field_info<Layout, TAG_s_dq>, field_info<Layout, TAG_s_dkv>>; \
        bytes_s = sizeof(typename Info::type) * bm * bn; \
    } \
    if constexpr (has_p_dkv<Layout>::value) { \
        using Info = field_info<Layout, TAG_p_dkv>; \
        bytes_p = sizeof(typename Info::type) * bm * bn; \
    } \
    bytes_sp = (bytes_s > bytes_p) ? bytes_s : bytes_p; \
    if (bytes_sp > 0) { \
        printf("   S/P (reuse) : %6zu B (%5.2f KB)  [S=%zu P=%zu]\n", bytes_sp, bytes_sp / 1024.0f, bytes_s, bytes_p); \
    } \
    if constexpr (has_dOV_dq<Layout>::value || has_dOV_dkv<Layout>::value) { \
        using Info = std::conditional_t<has_dOV_dq<Layout>::value, field_info<Layout, TAG_dOV_dq>, field_info<Layout, TAG_dOV_dkv>>; \
        bytes_dOV = sizeof(typename Info::type) * bm * bn; \
    } \
    if constexpr (has_dS_dq<Layout>::value || has_dS_dkv<Layout>::value) { \
        using Info = std::conditional_t<has_dS_dq<Layout>::value, field_info<Layout, TAG_dS_dq>, field_info<Layout, TAG_dS_dkv>>; \
        bytes_dS = sizeof(typename Info::type) * bm * bn; \
    } \
    bytes_dovs = (bytes_dOV > bytes_dS) ? bytes_dOV : bytes_dS; \
    if (bytes_dovs > 0) { \
        printf("   dOV/dS(reuse):%6zu B (%5.2f KB)  [dOV=%zu dS=%zu]\n", bytes_dovs, bytes_dovs / 1024.0f, bytes_dOV, bytes_dS); \
    } \
    if constexpr (has_lse<Layout>::value && has_row_dot<Layout>::value) { \
        using InfoLSE = field_info<Layout, TAG_lse>; \
        using InfoDot = field_info<Layout, TAG_row_dot>; \
        bytes_stats = sizeof(typename InfoLSE::type) * bm + sizeof(typename InfoDot::type) * bm; \
        printf("   lse+rowdot  : %6zu B (%5.2f KB)\n", bytes_stats, bytes_stats / 1024.0f); \
    } \
    if constexpr (has_dQ_dq<Layout>::value) { \
        using Info = field_info<Layout, TAG_dQ_dq>; \
        bytes_dq = sizeof(typename Info::type) * Config::DQ::BLOCK_M * D; \
        printf("   dQ          : %6zu B (%5.2f KB)\n", bytes_dq, bytes_dq / 1024.0f); \
    } \
    if constexpr (has_dK_dkv<Layout>::value && has_dV_dkv<Layout>::value) { \
        using InfoK = field_info<Layout, TAG_dK_dkv>; \
        bytes_dkv = sizeof(typename InfoK::type) * Config::DKV::BLOCK_N * D * 2; \
        printf("   dK+dV       : %6zu B (%5.2f KB)\n", bytes_dkv, bytes_dkv / 1024.0f); \
    } \
    printf("   -----------------------------------------\n"); \
    printf("   TOTAL       : %6zu B (%5.2f KB)\n", smem_total, smem_total / 1024.0f); \
    const int blk_by_thr = MAX_THREADS_PER_SM / Config::THREADS_PER_BLOCK; \
    const int blk_by_smem = (smem_total > 0) ? (MAX_SMEM_PER_SM / smem_total) : MAX_THREAD_BLOCK_PER_SM; \
    const int blk_per_sm = min(min(blk_by_thr, blk_by_smem), MAX_THREAD_BLOCK_PER_SM); \
    const int act_warps  = blk_per_sm * Config::WARPS_PER_BLOCK; \
    const float occ      = static_cast<float>(act_warps) / MAX_WARPS_PER_SM; \
    printf(" Occupancy:\n"); \
    printf("   Blocks/SM   : %d  (thr:%d  smem:%d  hw:%d)\n", \
           blk_per_sm, blk_by_thr, blk_by_smem, MAX_THREAD_BLOCK_PER_SM); \
    printf("   Warps/SM    : %d / %d\n", act_warps, MAX_WARPS_PER_SM); \
    printf("   Occupancy   : %5.1f%%\n", occ * 100.0f); \
    printf("======================================================================\n\n"); \
} while(0)

#else
    #define __ASM_DEBUG_BEGIN(STG, CTX) ((void)0)
    #define __ASM_DEBUG_END(STG, CTX)   ((void)0)
    #define __CHECK_INIT(FIELD_TAG, EXPECTED_VAL, VALID_ROWS)
    #define __CHECK_ERRORS(STG, VM, VN, SC, WID, LID, TID)
    #define __PRINT_MATRIX(STG, VM, VN, SC, WID, LID, TID, TILE_IDX)
    #define __PRINT_RESULT(FIELD_TAG, VLEN, TILE_IDX)
    #define __PRINT_OCCUPANCY_FORWARD(D, M, N, B, H_Q, H_K, grid, block)
    #define __PRINT_OCCUPANCY_BACKWARD(D, M, N, B, H_Q, H_K, grid, block, grid_dq, grid_dkv)
#endif