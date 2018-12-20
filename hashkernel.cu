#include <stdint.h>
#include <chrono>
#include "hashtable.h"
#include "hashfunction.h"
#include "hashkernel.h"

__constant__ uint32_t c_crc_x64_32_tab[HASH_TAB_SIZE];


template <typename T>
__host__ int getnumdistinctkeys(T* h_key_columns, int num_key_columns, int num_key_rows)
{
    return num_key_rows; //for now return number of rows
}


template <typename T>
__device__ int comparekeyrows(T* d_key_columns, int num_key_columns, int num_key_rows,
                     int a, int b)
{
    int equal = 1;
    for (int i = 0; i < num_key_columns; i++) {
        if (d_key_columns[i * num_key_rows + a] != d_key_columns[i * num_key_rows + b])
            return 0;
    }
    return equal;
}


template <typename T>
__global__ void groupbykernel(T* d_key_columns, int num_key_columns, int num_key_rows,
                              T* d_value_columns, int num_value_columns, int num_value_rows,
                              reduction_op reduct_ops[], int num_ops,
                              hashbucket<T>* d_hashtable, int hash_table_rows, int* d_num_unique_keys)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < num_key_rows) {

      int bucket_index = crc_x64_32_hash<T>(d_key_columns, num_key_columns, num_key_rows, row) % hash_table_rows;
      int old_key_row, current_key_row;

      int tryagain = 1;
      while (tryagain) {

        current_key_row = d_hashtable[bucket_index].key_row;
        if (current_key_row == EMPTYMARKER) {

          old_key_row = atomicCAS(&d_hashtable[bucket_index].key_row, current_key_row, row);

          if (old_key_row != current_key_row) {
            current_key_row = old_key_row;
          } else {
            tryagain = 0; //key was inserted, proceed to update reduction fields
            current_key_row = row;
            atomicAdd(d_num_unique_keys, 1); //update count of unique keys
          }
        }

        if (current_key_row != row) {
          //compare rows
          if (comparekeyrows<T>(d_key_columns, num_key_rows, num_key_columns, current_key_row, row)) {
            tryagain = 0; //found matching bucket, proceed to update reduction fields
          } else {
            tryagain = 1; //collision, try next bucket
            bucket_index = (bucket_index + 1) % hash_table_rows;
          }
        }
        __syncthreads();
      }

      //update reduction fields
      for (int i = 0; i < num_ops; i++) {
        if (reduct_ops[i] == max) {
          atomicMax(&d_hashtable[bucket_index].max, d_value_columns[i * num_value_rows + row]);
        } else if (reduct_ops[i] == min) {
          atomicMin(&d_hashtable[bucket_index].min, d_value_columns[i * num_value_rows + row]);
        } else if (reduct_ops[i] == sum) {
          atomicAdd(&d_hashtable[bucket_index].sum, d_value_columns[i * num_value_rows + row]);
        } else if (reduct_ops[i] == count) {
          atomicAdd(&d_hashtable[bucket_index].count, 1);
        }
      }

  }
}


template <typename T>
__global__ void getouputdatakernel(T* d_output_keys, int num_key_columns, int num_key_rows,
                                   T* d_output_values, int num_value_columns, int num_value_rows,
                                   hashbucket<T>* d_hashtable, int num_unique_keys, int hash_table_rows,
                                   reduction_op reduct_ops[], int num_ops, T* d_key_columns)
{
    int output_row = threadIdx.x + blockIdx.x * blockDim.x;

    if (output_row < num_unique_keys) {

        int scan_size = (hash_table_rows / num_unique_keys);
        int num_scan_rows = scan_size;
        if ((output_row = num_unique_keys - 1) && (num_key_rows % num_unique_keys)) {
            num_scan_rows += 1;
        }

        int start_row = output_row * scan_size;
        hashbucket<T> bucket;

        for (int i = 0; i < num_scan_rows; i++) {
            start_row += i;
            bucket = d_hashtable[start_row];
            if (bucket.key_row != EMPTYMARKER) {

                //copy row
                for (int j = 0; j < num_key_columns; j++) {
                    d_output_keys[j * num_unique_keys + output_row] = d_key_columns[j * num_key_rows + bucket.key_row];
                }

                //copy reduction values
                for (int k = 0; k < num_ops; k++) {
                    if (reduct_ops[k] == max) {
                        d_output_values[k * num_unique_keys + output_row] = bucket.max;
                    } else if (reduct_ops[k] == min) {
                        d_output_values[k * num_unique_keys + output_row] = bucket.min;
                    } else if (reduct_ops[k] == sum) {
                        d_output_values[k * num_unique_keys + output_row] = bucket.sum;
                    } else if (reduct_ops[k] == count) {
                        d_output_values[k * num_unique_keys + output_row] = bucket.count;
                    }
                }
            }
        }
    }
}


template <typename T>
__host__ struct output_data groupby(T* h_key_columns, int num_key_columns, int num_key_rows,
                                    T* h_value_columns, int num_value_columns, int num_value_rows,
                                    reduction_op ops[], int num_ops)
{

  //get number of unique keys
  int* h_num_unique_keys, d_num_unique_keys;
  cudaMalloc((void **) &d_num_unique_keys, sizeof(int));
  int hash_table_rows = getnumdistinctkeys<T>(h_key_columns, num_key_columns, num_key_rows);

  //allocate memory for hash table on device
  hashbucket<T>* d_hashtable;
  hashtablesize = hash_table_rows * sizeof(hashbucket<T>);
  cudaMalloc((void **) &d_hashtable, hashtablesize);

  //initialize hash table
  init_hash_table<T>(d_hashtable, hash_table_rows);

  //transfer keys and values data to device
  T* d_key_columns, d_value_columns;
  int num_key_pitch, num_value_pitch;

  int key_data_size = num_key_rows * num_key_columns * sizeof(T);
  cudaMalloc((void **)&d_key_columns, key_data_size);
  cudaMemcpy(d_key_columns, h_key_columns, key_data_size, cudaMemcpyHostToDevice);

  int value_data_size = num_value_rows * num_value_columns * sizeof(T);
  cudaMalloc((void **)&d_value_columns, value_data_size);
  cudaMemcpy(d_value_columns, h_value_columns, value_data_size, cudaMemcpyHostToDevice);

  //copy hash key tab to constant memory
  cudaMemcpytoSymbol(c_crc_x64_32_tab, crc_x64_32_tab, HASH_TAB_SIZE * sizeof(uint32_t));

  //launch reduction kernel
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((num_key_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  groupbykernel<T><<<dimGrid, dimBlock>>>(d_key_columns, num_key_columns, num_key_rows,
                                          d_value_columns, num_value_columns, num_value_rows,
                                          reduct_ops, num_ops,
                                          d_hashtable, hash_table_rows, d_num_unique_keys);
  cudaDeviceSynchronize();

  //copy number of unique keys from device memory
  cudaMemcpy(h_num_unique_keys, d_num_unique_keys, sizeof(int), cudaMemcpyDeviceToHost);

  //allocate space on host memory for output keys and output values
  int output_key_size = h_num_unique_keys * num_key_columns * sizeof(T);
  int output_values_size = h_num_unique_keys * num_value_columns * sizeof(T);

  T* h_output_keys, h_output_values;
  cudaMallocHost(&h_output_keys, output_key_size);
  cudaMallocHost(&h_output_values, output_values_size);

  T* d_output_keys, d_output_values;
  cudaMalloc((void **) &d_output_keys, output_key_size);
  cudaMalloc((void **) &d_output_values, output_values_size);

  //launch kernel to summarize results in output format
  dim3 dimGrid((h_num_unique_keys + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  getouputdatakernel<T><<<dimGrid, dimBlock>>>(d_output_keys, num_key_columns, num_key_rows,
                                               d_output_values, num_value_columns, num_value_rows
                                               d_hashtable, h_num_unique_keys, hash_table_rows,
                                               reduct_ops, num_ops, d_key_columns);

  //copy results back to host
  cudaMemcpy(h_output_keys, d_output_keys, output_key_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output_values, d_output_values, output_values_size, cudaMemcpyDeviceToHost);

  struct output_data<T> output;
  output.keys = h_output_keys;
  output.values = h_output_values;
  output.unique_keys = h_num_unique_keys;

  //free device memory
  cudaFree(d_num_unique_keys);
  cudaFree(d_hashtable);
  cudaFree(d_key_columns);
  cudaFree(d_value_columns);
  cudaFree(d_output_keys);
  cudaFree(d_output_values);

  return output;
}
