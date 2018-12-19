#include <stdint.h>
#include "hashtable.h"
#include "hashfunction.h"
#include "hashkernel.h"

template <typename T>
__global__ void groupbykernel(T* d_key_columns, int num_key_columns, int num_key_rows,
                              T* d_value_columns, int num_value_columns, int num_value_rows,
                              reduction_op reduct_ops[], int num_ops,
                              hashbucket<T>* d_hashtable, int hash_table_rows, uint32_t* d_crc_x64_32_result)
{
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row < num_key_rows) {
      uint32_t hashkey = d_crc_x64_32_result[row];
      int hash_map_index = hashkey % hash_table_rows;

      uint32_t old_hashkey;

      int tryagain = 1;
      while (tryagain) {
        if (d_hashtable[hash_map_index].row_index_count == 0) {

          old_hashkey = atomicAdd(&d_hashtable[hash_map_index].hashkey, hashkey);

          if (old_hashkey != hashkey) tryagain = 1; //lost race, try next bucket
          else {
            atomicAdd(&d_hashtable[hash_map_index].row_index_count, 1);
            tryagain = 0;
          }

        } else if (d_hashtable[hash_map_index].hashkey != hashkey) {
          tryagain = 1; //collision try next bucket,
          hash_map_index = (hash_map_index + 1) % hash_table_rows;
        } else {
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
__device__ int acquire_bucket_lock(uint32_t* hash_key_address, uint32_t  hash_value)
{
  int* address_as_int = (int *) d_histobin;
  int old = *d_histobin;
  int current;

	do {
    if (old < UINT8_MAXIMUM) {
      current = old;
      old = atomicCAS(address_as_int, current, current + 1);
    } else break;
	} while (current != old);
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

  //get number of distinct keys
  int hash_table_rows = get_num_distinct_keys<T>(h_key_columns, num_key_columns, num_key_rows);

  //create hash table
  HashTable<T>::set_hashtable_rows(hash_table_rows);
  HashTable<T>::init_hash_table();

  //transfer hash table from host to device
  hashbucket<T>* d_hashtable;
  hashtablesize = hash_table_rows * sizeof(hashbucket<T>);
  cudaMalloc((void **) &d_hashtable, hashtablesize);
  cudaMemcpy(d_hashtable, HashTable<T>::h_hashtable, hashtablesize, cudaMemcpyHostToDevice);

  //transfer keys and values data to device
  int *d_key_columns, *d_value_columns;
  int num_key_pitch, num_value_pitch;

  cudaMallocPitch(&d_key_columns, &num_key_pitch, num_key_columns * sizeof(T), num_key_rows);
  cudaMemcpy2D(d_key_columns, num_key_pitch, h_key_columns, num_key_pitch,
               num_key_columns, num_key_rows, cudaMemcpyHostToDevice);


  cudaMallocPitch(&d_value_columns, &num_value_pitch, num_value_columns * sizeof(T), num_value_rows);
  cudaMemcpy2D(d_value_columns, num_value_pitch, h_value_columns, num_value_pitch,
               num_value_columns, num_value_rows, cudaMemcpyHostToDevice);


  //get hash key table
  uint32_t seed = 7;
  uint32_t* h_crc_x64_32_result, d_crc_x64_32_result;
  h_crc_x64_32_result = (uint32_t *) malloc(num_key_rows * sizeof(uint32_t));
  cudaMalloc((void**)&d_crc_x64_32_result, num_key_rows * sizeof(uint32_t));

  crc_x64_32_host<T>(d_key_columns, num_key_columns, num_key_rows,
                     h_crc_x64_32_result, d_crc_x64_32_result, seed);

  //launch kernel
  numBlocks = ceil(num_key_rows / (float) BLOCK_SIZE);

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);
  groupbykernel<T><<<dimGrid, dimBlock>>>(d_key_columns, num_key_columns, num_key_rows,
                                          d_value_columns, num_value_columns, num_value_rows,
                                          reduct_ops, num_ops,
                                          d_hashtable, hash_table_rows, d_crc_x64_32_result);
  cudaDeviceSynchronize();


  //get result back
  cudaMemcpy(HashTable<T>::h_hashtable, d_hashtable, hashtablesize, cudaMemcpyDeviceToHost);

  //transform data to output keys and values format


  //delete hash HashTable
  HashTable<T>::delete_hash_table();

  //free heap memory on host and device
  free(h_crc_x64_32_result);
  cudaFree(d_crc_x64_32_result);
  cudaFree(d_hashtable);
  cudaFree(d_key_columns);
  cudaFree(d_value_columns);
}
