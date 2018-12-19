#ifndef _HASH_TABLE_H_
#define _HASH_TABLE_H_

#include <iostream>
#include <stdint.h>

using namespace std;

#define EMPTYMARKER -1;
#define BLOCK_SIZE 1024;

template <datatype T>
struct hashbucket {
  int key_row;
  int row_count;
  T max;
  T min;
  T sum;
  int count;
};

template <typename T>
__global__ void init_hash_kernel(hashbucket<T>* d_hashtable);

template <typename T>
__host__ void init_hash_table(hashbucket<T>* d_hashtable, const int hash_table_rows);







#endif //_HASH_TABLE_H_
