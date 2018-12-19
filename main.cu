#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "hashtable.h"
#include "hashkernel.h"


using namespace std;

void main()
{
  //generate input data
  int *h_key_cols[];
  int n_key_cols;
  int n_key_rows;
  int *h_value_cols[];
  int n_value_cols;
  int n_value_rows;
  reduction_op reduct_ops[];
  int n_ops;
  int *out_keys[];
  int *out_values[];



}
