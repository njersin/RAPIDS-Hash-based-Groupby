#ifndef _HASH_FUNCTION_H_
#define _HASH_FUNCTION_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#define HASH_TAB_SIZE 256
#define BLOCK_SIZE 1024

extern __constant__ uint32_t  crc_x64_32_tab[HASH_TAB_SIZE];

template <typename T>
__device__ uint8_t retrieve_byte(const T* d_key_columns,
    const int num_key_columns,
    const int num_key_rows,
    const size_t idx,
    const size_t current_byte);


template <typename T>
__global__ void crc_x64_32_hash(const T* d_key_columns,
    const int num_key_columns,
    const int num_key_rows,
    uint32_t* d_crc_x64_32_result,
    const uint32_t seed);


template <typename T>
__host__ void crc_x64_32_host(const T* d_key_columns,
                              const int num_key_columns,
                              const int num_key_rows,
                              uint32_t* h_crc_x64_32_result,
                              uint32_t* d_crc_x64_32_result,
                              const uint32_t seed);

#endif //_HASH_FUNCTION_H_
