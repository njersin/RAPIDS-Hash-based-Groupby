#ifndef _HASH_TABLE_H_
#define _HASH_TABLE_H_

#include <iostream>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define BLOCK_SIZE 1024
#define EMPTYMARKER -1

using namespace std;

template <typename T>
struct hashbucket {
  int key_row;
  T max;
  T min;
  T sum;
  int count;
};

template <typename T>
__global__ void init_hash_kernel(hashbucket<T>* d_hashtable, const int hash_table_rows);

template <typename T>
__host__ void init_hash_table(hashbucket<T>* d_hashtable, const int hash_table_rows);


#endif //_HASH_TABLE_H_
