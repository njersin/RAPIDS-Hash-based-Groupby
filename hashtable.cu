#include "hashtable.cuh"

template <typename T>
__global__ void init_hash_kernel(hashbucket<T>* d_hashtable, const int hash_table_rows)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < hash_table_rows) {
        d_hashtable[row].key_row = EMPTYMARKER;
        d_hashtable[row].max = 0;
        d_hashtable[row].min = 0;
        d_hashtable[row].sum = 0;
        d_hashtable[row].count = 0;
    }
}


template <typename T>
__host__ void init_hash_table(hashbucket<T>* d_hashtable, const int hash_table_rows)
{
    //launch kernel
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid((hash_table_rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    init_hash_kernel<T><<<dimGrid, dimBlock>>>(d_hashtable, hash_table_rows);
    cudaDeviceSynchronize();
    return;
}
