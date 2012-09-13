  // Copyright 2012 Justus Sagemüller.
  // This file is part of the Hilbert library.
   //This library is free software: you can redistribute it and/or modify
  // it under the terms of the GNU General Public License as published by
 //  the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
   //This library is distributed in the hope that it will be useful,
  // but WITHOUT ANY WARRANTY; without even the implied warranty of
 //  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
  // You should have received a copy of the GNU General Public License
 //  along with this library.  If not, see <http://www.gnu.org/licenses/>.


#ifndef CUDA_ACCELLERATION_FOR_DOUBLE_HILBERTSPACES
#define CUDA_ACCELLERATION_FOR_DOUBLE_HILBERTSPACES

//#include <math.h>
//#include <cuda_runtime.h>
#include "cublas_v2.h"

typedef struct {
  const cublasHandle_t* cublashandle;
  int vect_dimension;
  double* vector;
  cudaError_t cudastat;
  cublasStatus_t cublasstat;
}cudaDoubleHilbertspcVectHandle;

int
  cudaDoubleHilbertspcVect_bad
  ( const cudaDoubleHilbertspcVectHandle* handle );

void
  cudaDoubleHilbertspcVect_printerrmsg
  ( const cudaDoubleHilbertspcVectHandle* handle );

cudaDoubleHilbertspcVectHandle
  unassigned_cudaDoubleHilbertspcVectHandle
  ( const cublasHandle_t* cublashandle          );

cudaDoubleHilbertspcVectHandle
  undefined_cudaDoubleHilbertspcVect
  ( unsigned dimension
  , const cublasHandle_t* cublashandle          );

cudaDoubleHilbertspcVectHandle
  new_cudaDoubleHilbertspcVect
  ( const double* source     //on host
  , unsigned dimension
  , const cublasHandle_t* cublashandle     );

void
  get_cudaDoubleHilbertspcVect
  ( double* target
  , cudaDoubleHilbertspcVectHandle* handle );

void
  delete_cudaDoubleHilbertspcVect
  ( cudaDoubleHilbertspcVectHandle* handle );

cudaDoubleHilbertspcVectHandle
  copy_cudaDoubleHilbertspcVect
  ( cudaDoubleHilbertspcVectHandle* handle );

void
  axpy_cudaDoubleHilbertspcVect
  ( double alpha
  , cudaDoubleHilbertspcVectHandle* xhandle    //const as far as the vector is concerned
  , cudaDoubleHilbertspcVectHandle* yhandle );

double
  dot_cudaDoubleHilbertspcVect
  ( cudaDoubleHilbertspcVectHandle* xhandle   //const vector
  , cudaDoubleHilbertspcVectHandle* yhandle   //const vector
  );



#endif