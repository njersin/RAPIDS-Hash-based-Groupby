#include "hashtable.h"

template <typename T>
void HashTable<T>::init_hash_table()
{
  h_hashtable = new hashbucket<T>[hash_table_rows];
  for (int i = 0; i < hash_table_rows; i++) {
    h_hashtable[i].hashkey = EMPTYMARKER;
    h_hashtable[i].row_input_index = -1;
    h_hashtable[i].row_index_count = 0;
    h_hashtable[i].state = empty;
    h_hashtable[i].max = 0;
    h_hashtable[i].min = 0;
    h_hashtable[i].sum = 0;
    h_hashtable[i].count = 0;
  }
}


template <typename T>
void HashTable<T>::delete_hash_table()
{
  for (int i = 0; i < hash_table_rows; i++) {
    if (h_hashtable[i].rowindexlist) delete[] h_hashtable[i].rowindexlist;
  }
  delete[] h_hashtable;
}

template <typename T>
void HashTable<T>::set_hashtable_rows(int num_rows)
{
  hash_table_rows = num_rows;
}
