#ifndef _HASH_KERNEL_H_
#define _HASH_KERNEL_H_

enum reduction_op {max, min, sum, count};
enum bucket_state {empty, in_use, occupied};

_global__ groupbykernel(*d_key_cols[], n_key_rows, n_key_cols,
                        *d_value_cols[], n_value_rows, n_value_cols,
                        reduct_ops[], hashbucket<T>* d_hashtable, num_unique_keys);







#endif //_HASH_KERNEL_H_
