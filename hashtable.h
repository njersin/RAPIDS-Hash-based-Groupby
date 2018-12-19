#ifndef _HASH_TABLE_H_
#define _HASH_TABLE_H_

#include <iostream>
#include <stdint.h>

using namespace std;
const int EMPTYMARKER = 0;

enum bucket_state {empty, in_use, occupied};

template <datatype T>
struct hashbucket {
  uint32_t hashkey;
  int row_input_index;
  int row_index_count;
  int state;
  T max;
  T min;
  T sum;
  int count;
};


template <typename T>
class HashTable
{
  public:
    int hash_table_rows;
    void set_hashtable_rows(int num_rows);
    hashbucket<T>* h_hashtable = NULL;
    void init_hash_table();
    void delete_hash_table();





}
#endif //_HASH_TABLE_H_
