#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cpuGroupby.h>
#include "hashkernel.h"

using namespace std;

int main(int argc, char *argv[])
{

  using Time = std::chrono::high_resolution_clock;
  using fsec = std::chrono::duration<float>;

  int num_rows = 100000;
  int num_key_cols = 2;
  int num_val_cols = 3;
  int num_distinct_keys = 3;
  if (argc == 2){
    num_rows = atoi(argv[1]);
  }else if(argc ==4){
    num_rows = atoi(argv[1]);
    num_key_cols = atoi(argv[2]);
    num_val_cols = atoi(argv[3]);
  }else if(argc ==5){
    num_rows = atoi(argv[1]);
    num_key_cols = atoi(argv[2]);
    num_val_cols = atoi(argv[3]);
    num_distinct_keys = atoi(argv[4]);
  }
  // Setting up the CPU groupby
  cpuGroupby slowGroupby(num_key_cols, num_val_cols, num_rows);

  slowGroupby.fillRand(num_distinct_keys, num_rows);

  int *original_key_columns;
  cudaMallocHost((void**)&original_key_columns, sizeof(int)*num_key_cols*num_rows);
  int *original_value_columns;
  cudaMallocHost((void**)&original_value_columns, sizeof(int)*num_val_cols*num_rows);
  std::copy(slowGroupby.key_columns, slowGroupby.key_columns + num_key_cols*num_rows, original_key_columns);
  std::copy(slowGroupby.value_columns, slowGroupby.value_columns + num_val_cols*num_rows, original_value_columns);

  auto start = Time::now();

  slowGroupby.groupby();

  auto end = Time::now();
  fsec cpu_duration = end - start;


  //run gpu kernel
  start = Time::now();

  int num_ops = 4;
  reduction_op ops[4] = {max, min, sum, count};

  struct output_data<int> gpu_output;
  gpu_output = groupby<int>(original_key_columns, num_key_cols, num_rows,
                            original_value_columns, num_val_cols, num_rows,
                            ops, num_ops)
  end = Time::now();

  //print gpu results
  slowGroupby.printGPUResults(gpu_output.keys, gpu_output.values);

  fsec gpu_duration = end - start;

  cout << "CPU time: " << cpu_duration.count() << " s" << endl;
  cout << "GPU time: " << gpu_duration.count() << " s" << endl;

  //validate gpu results
  slowGroupby.validGPUResult(gpu_output.keys, gpu_output.values, gpu_output.unique_keys);

  cudaFreeHost(original_value_columns);
  cudaFreeHost(original_key_columns);
  cudaFreeHost(gpu_output.keys);
  cudaFreeHost(gpu_output.values);
  return 0;

}
