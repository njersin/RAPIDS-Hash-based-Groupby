#ifndef _HASH_KERNEL_H_
#define _HASH_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include "hashtable.cuh"
#include "hashfunction.cuh"

#define EMPTYMARKER -1

enum reduction_op {max_op, min_op, sum, count};

template <typename T>
struct output_data {
    T* keys;
    T* values;
    int unique_keys;
};

template <typename T>
__host__ int getnumdistinctkeys(T* h_key_columns, int num_key_columns, int num_key_rows);


template <typename T>
__device__ int comparekeyrows(T* d_key_columns, int num_key_columns, int num_key_rows,
                                int a, int b);


template <typename T>
__global__ void groupbykernel(T* d_key_columns, int num_key_columns, int num_key_rows,
                              T* d_value_columns, int num_value_columns, int num_value_rows,
                              reduction_op reduct_ops[], int num_ops,
                              hashbucket<T>* d_hashtable, int hash_table_rows, int* d_unique_keys);

template <typename T>
__global__ void getouputdatakernel(T* d_output_keys, int num_key_columns, int num_key_rows,
                                   T* d_output_values, int num_value_columns, int num_value_rows,
                                   hashbucket<T>* d_hashtable, int num_unique_keys, int hash_table_rows,
                                   reduction_op reduct_ops[], int num_ops, T* d_key_columns);


template <typename T>
__host__ struct output_data<T> groupby(T* h_key_columns, int num_key_columns, int num_key_rows,
                                    T* h_value_columns, int num_value_columns, int num_value_rows,
                                    reduction_op ops[], int num_ops);


#endif //_HASH_KERNEL_H_
