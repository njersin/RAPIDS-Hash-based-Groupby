#include <stdint.h>
#include "hashtable.h"
#include "hashfunction.h"
#include "hashkernel.h"

__constant__ uint32_t c_crc_x64_32_tab[HASH_TAB_SIZE];


template <typename T>
__global__ void groupbykernel(T* d_key_columns, int num_key_columns, int num_key_rows,
                              T* d_value_columns, int num_value_columns, int num_value_rows,
                              reduction_op reduct_ops[], int num_ops,
                              hashbucket<T>* d_hashtable, int hash_table_rows, int* d_unique_keys)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < num_key_rows) {
      uint32_t hashkey = d_crc_x64_32_result[row];
      int hash_map_index = hashkey % hash_table_rows;

      int old_row;

      int tryagain = 1;
      while (tryagain) {
        if (d_hashtable[hash_map_index].row_index_count == EMPTYMARKER) {

          old_row = atomicCAS(&d_hashtable[hash_map_index].row_index_count, EMPTYMARKER, row);

          if (old_row != row) tryagain = 1; //lost race, try next bucket
          else {
            //compare keys if match increment count else move to next bucket
            atomicAdd(&d_hashtable[hash_map_index].row_index_count, 1);
            tryagain = 0;
          }

        } else if (d_hashtable[hash_map_index].hashkey != hashkey) {
          tryagain = 1; //collision try next bucket
          hash_map_index = (hash_map_index + 1) % hash_table_rows;
        } else {
          tryagain = 0; //found matching bucket, update row count and exit loop
          atomicAdd(&d_hashtable[hash_map_index].row_index_count, 1);
        }
        __syncthreads();
      }

      //add row to rowindexlist
      for (int i = 0; i < num_ops; i++) {
        if (reduct_ops[i] == max) {
          atomicMax(&d_hashtable[hash_map_index].max, d_value_columns[i * num_value_rows + row]);
        } else if (reduct_ops[i] == min) {
          atomicMin(&d_hashtable[hash_map_index].min, d_value_columns[i * num_value_rows + row]);
        } else if (reduct_ops[i] == sum) {
          atomicAdd(&d_hashtable[hash_map_index].sum, d_value_columns[i * num_value_rows + row]);
        } else if (reduct_ops[i] == count) {
          atomicAdd(&d_hashtable[hash_map_index].count, 1);
        }
      }

  }
}


template <typename T>
int get_num_distinct_keys(T* h_key_columns[], int num_key_columns, int num_key_rows)
{
  return num_key_rows; //for now return number of keys
}


template <typename T>
void groupby(T* h_key_columns[], int num_key_columns, int num_key_rows,
             T* h_value_columns[], int num_value_columns, int num_value_rows,
             reduction_op ops[], int num_ops,
             T* output_keys[], T* output_values[])
{

  //get number of unique keys
  int* h_unique_keys, d_unique_keys;
  int hash_table_rows = get_num_distinct_keys<T>(h_key_columns, num_key_columns, num_key_rows);

  //allocate memory for hash table on device
  hashbucket<T>* d_hashtable;
  hashtablesize = hash_table_rows * sizeof(hashbucket<T>);
  cudaMalloc((void **) &d_hashtable, hashtablesize);

  //initialize hash table
  init_hash_table<T>(d_hashtable, hash_table_rows);

  //transfer keys and values data to device
  T* d_key_columns, d_value_columns;
  int num_key_pitch, num_value_pitch;

  cudaMallocPitch(&d_key_columns, &num_key_pitch, num_key_columns * sizeof(T), num_key_rows);
  cudaMemcpy2D(d_key_columns, num_key_pitch, h_key_columns, num_key_pitch,
               num_key_columns, num_key_rows, cudaMemcpyHostToDevice);

  cudaMallocPitch(&d_value_columns, &num_value_pitch, num_value_columns * sizeof(T), num_value_rows);
  cudaMemcpy2D(d_value_columns, num_value_pitch, h_value_columns, num_value_pitch,
               num_value_columns, num_value_rows, cudaMemcpyHostToDevice);

  //copy hash keytab to constant memory
  cudaMemcpytoSymbol(c_crc_x64_32_tab, crc_x64_32_tab, HASH_TAB_SIZE * sizeof(uint32_t));

  //launch reduction kernel
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(ceil(num_key_rows / (float) BLOCK_SIZE), 1, 1);
  groupbykernel<T><<<dimGrid, dimBlock>>>(d_key_columns, num_key_columns, num_key_rows,
                                          d_value_columns, num_value_columns, num_value_rows,
                                          reduct_ops, num_ops,
                                          d_hashtable, hash_table_rows, d_unique_keys);
  cudaDeviceSynchronize();


  //get result back


  //transform data to output keys and values format


  //free device memory
  cudaFree(d_hashtable);
  cudaFree(d_key_columns);
  cudaFree(d_value_columns);
}
