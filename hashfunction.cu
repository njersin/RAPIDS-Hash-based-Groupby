#include "hashfunction.h"

template <typename T>
__device__ uint8_t retrieve_byte(const T* d_key_columns,
                                 const int num_key_columns,
                                 const int num_key_rows,
                                 const int row_idx,
                                 const int current_byte)
{
    int column_idx = current_byte / sizeof(T);
    int shift = 8 * (current_byte % sizeof(T));
    return (d_key_columns[column_idx * num_key_rows + row_idx] >> shift) & 0xff;
}


template <typename T>
__device__ uint32_t crc_x64_32_hash(const T* d_key_columns,
                                    const int num_key_columns,
                                    const int num_key_rows,
                                    const int row_idx)
{
    uint32_t hash = 0xffffffff;
    int message_size = sizeof(T) * num_key_columns;

    uint8_t message;

    for (int i = 0; i < message_size; ++i) {
        message = retrieve_byte<T>(d_key_columns, num_key_columns, num_key_rows, row_idx, i);
        hash = c_crc_x64_32_tab[(hash ^ message)] ^ (hash >> 8);
    }

    return hash ^ 0xffffffff;
}
