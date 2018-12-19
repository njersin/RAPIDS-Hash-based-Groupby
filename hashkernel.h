#ifndef _HASH_KERNEL_H_
#define _HASH_KERNEL_H_

#define EMPTYMARKER -1;

enum reduction_op {max, min, sum, count};

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
__global__ void getouputdatakernel(T* d_output_keys, int num_key_columns,
                                   T* d_output_values, int num_value_columns,
                                   int num_unique_keys);


template <typename T>
__host__ void groupby(T* h_key_columns, int num_key_columns, int num_key_rows,
                      T* h_value_columns, int num_value_columns, int num_value_rows,
                      reduction_op ops[], int num_ops,
                      T* output_keys, T* output_values);


#endif //_HASH_KERNEL_H_
