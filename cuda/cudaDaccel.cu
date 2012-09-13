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

#include "cudaDaccel.h"
#include <math.h>
#include <stdio.h>
#include "cublasaccel.h"

const int cuD_size = sizeof(double);

int cudaDoubleHilbertspcVect_bad(const cudaDoubleHilbertspcVectHandle* handle){
  return handle->cudastat   != cudaSuccess
      || handle->cublasstat != CUBLAS_STATUS_SUCCESS;
}
void cudaDoubleHilbertspcVect_printerrmsg(const cudaDoubleHilbertspcVectHandle* handle){
  if(handle->cudastat != cudaSuccess)
    printf("%s", cudaGetErrorString(handle->cudastat));
  if(handle->cublasstat != CUBLAS_STATUS_SUCCESS)
    print_cublas_status_message(handle->cublasstat);
}


cudaDoubleHilbertspcVectHandle
  unassigned_cudaDoubleHilbertspcVectHandle
              ( const cublasHandle_t* cublashandle ){
  cudaDoubleHilbertspcVectHandle handle;
  handle.cublashandle = cublashandle;
  handle.cudastat = cudaErrorMissingConfiguration;
  return handle;
};

cudaDoubleHilbertspcVectHandle undefined_cudaDoubleHilbertspcVect
              ( unsigned dimension
              , const cublasHandle_t* cublashandle          ) {
  cudaDoubleHilbertspcVectHandle handle;
  handle.cublashandle = cublashandle;
  handle.vect_dimension = dimension;
  handle.cudastat = cudaMalloc( (void**)&handle.vector
                              , dimension * sizeof(double) );
  return handle;
}


cudaDoubleHilbertspcVectHandle new_cudaDoubleHilbertspcVect
              ( const double* source
              , unsigned dimension
              , const cublasHandle_t* cublashandle     ) {
  cudaDoubleHilbertspcVectHandle handle;
  handle.cublashandle = cublashandle;
  handle.vect_dimension = dimension;
  handle.cudastat = cudaMalloc( (void**)&handle.vector
                               , dimension * cuD_size );
  if (handle.cudastat==cudaSuccess)
    handle.cublasstat = cublasSetVector( dimension, cuD_size
                                        , source
                                        , 1, handle.vector, 1 );
  return handle;
}


void get_cudaDoubleHilbertspcVect
              ( double* target
              , cudaDoubleHilbertspcVectHandle* handle ) {
  handle->cublasstat = cublasGetVector( handle->vect_dimension
                                      , cuD_size
                                      , handle->vector, 1
                                      , target, 1               );
}


void delete_cudaDoubleHilbertspcVect
              ( cudaDoubleHilbertspcVectHandle* handle ) {
  handle->cudastat = cudaFree(handle->vector);
}


cudaDoubleHilbertspcVectHandle copy_cudaDoubleHilbertspcVect
              ( cudaDoubleHilbertspcVectHandle* cpyhandle ) {
  cudaDoubleHilbertspcVectHandle handle = *cpyhandle;
  handle.cudastat = cudaMalloc( (void**)&handle.vector
                               , handle.vect_dimension * cuD_size );
  if (handle.cudastat==cudaSuccess)
    handle.cublasstat = cublasDcopy( *handle.cublashandle
                                   , handle.vect_dimension
                                   , cpyhandle->vector, 1
                                   , handle.vector, 1      );
  return handle;
}


void axpy_cudaDoubleHilbertspcVect
              ( double alpha
              , cudaDoubleHilbertspcVectHandle* xhandle
              , cudaDoubleHilbertspcVectHandle* yhandle ){
  // assert(xhandle->vect_dimension == yhandle->vect_dimension);
  xhandle->cublasstat = yhandle->cublasstat
     = cublasDaxpy( *xhandle->cublashandle
                  , xhandle->vect_dimension
                  , &alpha
                  , xhandle->vector, 1
                  , yhandle->vector, 1       );
}


double dot_cudaDoubleHilbertspcVect
              ( cudaDoubleHilbertspcVectHandle* xhandle
              , cudaDoubleHilbertspcVectHandle* yhandle ){
  double result;
    xhandle->cublasstat = yhandle->cublasstat
     = cublasDdot( *xhandle->cublashandle
                 , xhandle->vect_dimension
                 , xhandle->vector, 1
                 , yhandle->vector, 1
                 , &result                 );
  return result;
}




