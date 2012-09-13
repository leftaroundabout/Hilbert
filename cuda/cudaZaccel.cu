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

#include "cudaZaccel.h"

#include <math.h>
#include<stdio.h>
#include<assert.h>

#include "cublasaccel.h"
#include "cusparseaccel.h"

#include <cula.h>
#include <cula_status.h>
#include <cula_device.h>
#include <cula_lapack_device.h>


const unsigned cuZcplx_size = sizeof(cuDoubleComplex);

cusparseMatDescr_t simplesparsemat_descr_crt(){
  cusparseMatDescr_t d;
  cusparseCreateMatDescr(&d);
  cusparseSetMatType(d, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(d, CUSPARSE_INDEX_BASE_ZERO);  
  cusparseSetMatFillMode(d, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(d, CUSPARSE_DIAG_TYPE_NON_UNIT);
  return d;
}
cusparseMatDescr_t copy_sparsemat_descr_t(cusparseMatDescr_t cpy){
  cusparseMatDescr_t d;
  cusparseCreateMatDescr(&d);
  cusparseSetMatType(d, cusparseGetMatType(cpy));
  cusparseSetMatIndexBase(d, cusparseGetMatIndexBase(cpy));  
  cusparseSetMatFillMode(d, cusparseGetMatFillMode(cpy));
  cusparseSetMatDiagType(d, cusparseGetMatDiagType(cpy));
  return d;
}

int cudaCmplxDoubleHilbertspcVect_bad
              (const cudaCmplxDoubleHilbertspcVectHandle* handle){
  return handle->cudastat   != cudaSuccess
      || handle->cublasstat != CUBLAS_STATUS_SUCCESS;
}
void cudaCmplxDoubleHilbertspcVect_printerrmsg(const cudaCmplxDoubleHilbertspcVectHandle* handle){
  if(handle->cudastat != cudaSuccess)
    printf("CUDA error: %s\n", cudaGetErrorString(handle->cudastat));
  if(handle->cublasstat != CUBLAS_STATUS_SUCCESS)
    print_cublas_status_message(handle->cublasstat);
}
void cudaCmplxDoubleHilbertspcVect_resetStatus(cudaCmplxDoubleHilbertspcVectHandle* handle){
  handle->cudastat = cudaSuccess;
  handle->cublasstat = CUBLAS_STATUS_SUCCESS;
}

int cudaCmplxDoubleHilbertspcGenM_bad
              (const cudaCmplxDoubleHilbertspcGenMHandle* handle ) {
  return handle->cudastat   != cudaSuccess
      || handle->cublasstat != CUBLAS_STATUS_SUCCESS
      || handle->culastat   != culaNoError;
}
void cudaCmplxDoubleHilbertspcGenM_printerrmsg(const cudaCmplxDoubleHilbertspcGenMHandle* handle){
  if(handle->cudastat != cudaSuccess)
    printf("CUDA error: %s\n", cudaGetErrorString(handle->cudastat));
  if(handle->cublasstat != CUBLAS_STATUS_SUCCESS)
    print_cublas_status_message(handle->cublasstat);
  if(handle->culastat != culaNoError) {
    char* errorinfo = (char*)malloc(1024);
    culaGetErrorInfoString(handle->culastat, culaGetErrorInfo(), errorinfo, 1024);
    printf("CULA error: %s\nNAMELY{\n%s\n}\n", culaGetStatusString(handle->culastat), errorinfo);
    free(errorinfo);
  }
}
void cudaCmplxDoubleHilbertspcGenM_resetStatus(cudaCmplxDoubleHilbertspcGenMHandle* handle){
  handle->cudastat = cudaSuccess;
  handle->cublasstat = CUBLAS_STATUS_SUCCESS;
  handle->culastat = culaNoError;
}


int cudaCmplxDoubleHilbertspcSparseM_bad
              (const cudaCmplxDoubleHilbertspcSparseMHandle* handle ) {
  return handle->cudastat   != cudaSuccess
      || handle->cusparsestat != CUSPARSE_STATUS_SUCCESS;
}
void cudaCmplxDoubleHilbertspcSparseM_printerrmsg(const cudaCmplxDoubleHilbertspcSparseMHandle* handle){
  if(handle->cudastat != cudaSuccess)
    printf("%s", cudaGetErrorString(handle->cudastat));
  if(handle->cusparsestat != CUSPARSE_STATUS_SUCCESS)
    print_cusparse_status_message(handle->cusparsestat);
}
void cudaCmplxDoubleHilbertspcSparseM_resetStatus(cudaCmplxDoubleHilbertspcSparseMHandle* handle){
  handle->cudastat = cudaSuccess;
  handle->cusparsestat = CUSPARSE_STATUS_SUCCESS;
}


int cudaCmplxDoubleHilbertspcEigenbasisTransform_bad
              (const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle ) {
  return handle->cudastat   != cudaSuccess
      || handle->cublasstat != CUBLAS_STATUS_SUCCESS
      || handle->culastat   != culaNoError;
}
void cudaCmplxDoubleHilbertspcEigenbasisTransform_printerrmsg(const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle){
  if(handle->cudastat != cudaSuccess)
    printf("CUDA error: %s\n", cudaGetErrorString(handle->cudastat));
  if(handle->cublasstat != CUBLAS_STATUS_SUCCESS)
    print_cublas_status_message(handle->cublasstat);
  if(handle->culastat != culaNoError) {
    char* errorinfo = (char*)malloc(1024);
    culaGetErrorInfoString(handle->culastat, culaGetErrorInfo(), errorinfo, 1024);
    printf("CULA error: %s\nNAMELY{\n%s\n}\n", culaGetStatusString(handle->culastat), errorinfo);
    free(errorinfo);
  }
}
void cudaCmplxDoubleHilbertspcEigenbasisTransform_resetStatus(cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle){
  handle->cudastat = cudaSuccess;
  handle->cublasstat = CUBLAS_STATUS_SUCCESS;
  handle->culastat = culaNoError;
}




inline unsigned cudaCmplxDoubleHilbertspcGenM_datasize
              (const cudaCmplxDoubleHilbertspcGenMHandle* h) {
  if(h->matrix.matrixpacking==cuCmplxDouble_MATPACKING_TRIANG_UPPER
      ||h->matrix.matrixpacking==cuCmplxDouble_MATPACKING_TRIANG_LOWER )
    return h->matrix.packed_n_entries * cuZcplx_size;
   else
    return h->matrix.covardim * h->matrix.unpacked_cntvardim * cuZcplx_size;
}




cudaCmplxDoubleHilbertspcVectHandle unassigned_cudaCmplxDoubleHilbertspcVectHandle
              ( const cublasHandle_t* cublashandle ){
  cudaCmplxDoubleHilbertspcVectHandle handle;
  handle.cublashandle = cublashandle;
  handle.cudastat = cudaErrorMissingConfiguration;
  return handle;
};

cudaCmplxDoubleHilbertspcGenMHandle
  unassigned_cudaCmplxDoubleHilbertspcGenMHandle
              ( const cublasHandle_t* cublashandle          ) {
  cudaCmplxDoubleHilbertspcGenMHandle handle;
  handle.cublashandle = cublashandle;
  handle.cudastat = cudaErrorMissingConfiguration;
  return handle;
};


cudaCmplxDoubleHilbertspcSparseMHandle
  unassigned_cudaCmplxDoubleHilbertspcSparseMHandle
              ( const cusparseHandle_t* cusparsehandle          ) {
  cudaCmplxDoubleHilbertspcSparseMHandle handle;
  handle.cusparsehandle = cusparsehandle;
  handle.cudastat = cudaErrorMissingConfiguration;
  return handle;
};


cudaCmplxDoubleHilbertspcVectHandle undefined_cudaCmplxDoubleHilbertspcVect
              ( unsigned dimension
              , const cublasHandle_t* cublashandle ){
  cudaCmplxDoubleHilbertspcVectHandle handle;
  handle.cublashandle = cublashandle;
  handle.vect_dimension = dimension;
  handle.cudastat = cudaMalloc( (void**)&handle.vector
                              , dimension * cuZcplx_size );
  if (handle.cudastat==cudaSuccess)
    handle.cublasstat = CUBLAS_STATUS_SUCCESS;
  return handle;
};

cudaCmplxDoubleHilbertspcGenMHandle undefined_cudaCmplxDoubleHilbertspcGenM
              ( cuCmplxDoubleGenMatType mtype
              , cuCmplxDoubleGenMatPacking mtpacking
              , unsigned covardim
              , unsigned cntvardim
              , const cublasHandle_t* cublashandle ) {
  cudaCmplxDoubleHilbertspcGenMHandle handle;
  handle.cublashandle = cublashandle;
  handle.matrix.matrixtype = mtype;
  handle.matrix.matrixpacking = mtpacking;
  handle.matrix.covardim = covardim;
  if( mtpacking==cuCmplxDouble_MATPACKING_TRIANG_UPPER
       ||mtpacking==cuCmplxDouble_MATPACKING_TRIANG_LOWER ){
    assert(cntvardim==covardim);
    handle.matrix.packed_n_entries = covardim * (covardim + 1)/2;
   }else{
    handle.matrix.unpacked_cntvardim = cntvardim;
  }
  handle.cudastat = cudaMalloc( (void**)&handle.matrix.entries
                              , ( mtype==cuCmplxDouble_GENERIC_MATRIX
                                    ? covardim * cntvardim
                                    : handle.matrix.packed_n_entries )
                                * cuZcplx_size                        );
  if (handle.cudastat==cudaSuccess) {
    handle.cublasstat = CUBLAS_STATUS_SUCCESS;
    handle.culastat   = culaNoError;
  }
  return handle;
};



cudaCmplxDoubleHilbertspcVectHandle new_cudaCmplxDoubleHilbertspcVect
              ( const cuDoubleComplex* source
              , unsigned dimension
              , const cublasHandle_t* cublashandle     ) {
  cudaCmplxDoubleHilbertspcVectHandle handle;
  handle.cublashandle = cublashandle;
  handle.vect_dimension = dimension;
  handle.cudastat = cudaMalloc( (void**)&handle.vector
                              , dimension * cuZcplx_size );
  if (handle.cudastat==cudaSuccess)
    handle.cublasstat = cublasSetVector( dimension, cuZcplx_size
                                       , source
                                       , 1, handle.vector, 1     );
  return handle;
}

cudaCmplxDoubleHilbertspcGenMHandle new_cudaCmplxDoubleHilbertspcGenM
              ( const_cuCmplxDoubleGenMatEntries source
              , const cublasHandle_t* cublashandle     ) {
  cudaCmplxDoubleHilbertspcGenMHandle handle;
  handle.cublashandle = cublashandle;
  handle.matrix = source;
  handle.culastat = culaNoError;
  handle.cublasstat = CUBLAS_STATUS_SUCCESS;

  unsigned arraysz = cudaCmplxDoubleHilbertspcGenM_datasize(&handle);

  if ((handle.cudastat=cudaSuccess)==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.entries
                                , arraysz      );

  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMemcpy( handle.matrix.entries, source.entries
                                , arraysz
                                , cudaMemcpyHostToDevice                );
  return handle;
}


cudaCmplxDoubleHilbertspcSparseMHandle new_cudaCmplxDoubleHilbertspcSparseM
              ( const_cuCmplxDoubleSparseMatEntries source
              , const cusparseHandle_t* cusparsehandle     ) {
  cudaCmplxDoubleHilbertspcSparseMHandle handle;
  handle.cusparsehandle = cusparsehandle;
  handle.matrxdescript = simplesparsemat_descr_crt();
  handle.matrix = source;
  handle.inverseapphd = NULL;

  if ((handle.cudastat=cudaSuccess)==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.entries
                                , source.nnz * cuZcplx_size      );
  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.row
                                , (source.cntvardim+1) * sizeof(int) );
  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.col
                                , source.nnz * sizeof(int)    );

  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMemcpy( handle.matrix.entries, source.entries
                                , source.nnz * cuZcplx_size
                                , cudaMemcpyHostToDevice                );
  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMemcpy( handle.matrix.col, source.row   //temporarily copy to
                                , source.nnz * sizeof(int)       // yet-unused col index
                                , cudaMemcpyHostToDevice      );//  memory
  if (handle.cudastat==cudaSuccess)
    handle.cusparsestat = cusparseXcoo2csr( *handle.cusparsehandle  //move to row idx
                                          , handle.matrix.col      // memory as csr
                                          , handle.matrix.nnz, source.cntvardim
                                          , handle.matrix.row
                                          , CUSPARSE_INDEX_BASE_ZERO           );
  if (handle.cudastat==cudaSuccess && handle.cusparsestat == CUSPARSE_STATUS_SUCCESS)
    handle.cudastat = cudaMemcpy( handle.matrix.col, source.col
                                , source.nnz * sizeof(int)
                                , cudaMemcpyHostToDevice                );
  return handle;
}


void get_cudaCmplxDoubleHilbertspcVect
              ( cuDoubleComplex* target
              , cudaCmplxDoubleHilbertspcVectHandle* handle ) {
  handle->cublasstat = cublasGetVector( handle->vect_dimension
                                      , cuZcplx_size
                                      , handle->vector, 1
                                      , target, 1               );
}

void get_cudaCmplxDoubleHilbertspcGenM
              ( cuCmplxDoubleGenMatEntries target
              , cudaCmplxDoubleHilbertspcGenMHandle* handle ) {
  if (handle->cudastat==cudaSuccess)
    handle->cudastat = cudaMemcpy( target.entries, handle->matrix.entries
                                 , cudaCmplxDoubleHilbertspcGenM_datasize(handle)
                                 , cudaMemcpyDeviceToHost                                 );
}

void get_cudaCmplxDoubleHilbertspcSparseM
              ( cuCmplxDoubleSparseMatEntries target
              , cudaCmplxDoubleHilbertspcSparseMHandle* handle ) {

  int* coorowfmt;
  handle->cudastat = cudaMalloc( (void**)&coorowfmt
                               , (handle->matrix.cntvardim+1) * sizeof(int) );
  if (handle->cudastat==cudaSuccess)
    handle->cusparsestat = cusparseXcsr2coo( *handle->cusparsehandle
                                           , handle->matrix.row
                                           , handle->matrix.nnz, handle->matrix.cntvardim
                                           , coorowfmt
                                           , CUSPARSE_INDEX_BASE_ZERO           );
  if (handle->cudastat==cudaSuccess && handle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
    handle->cudastat = cudaMemcpy( target.row, coorowfmt
                                 , handle->matrix.nnz * sizeof(int)
                                 , cudaMemcpyDeviceToHost        );
  handle->cudastat = cudaFree(coorowfmt);

  if (handle->cudastat==cudaSuccess && handle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
    handle->cudastat = cudaMemcpy( target.col, handle->matrix.col
                                 , handle->matrix.nnz * sizeof(int)
                                 , cudaMemcpyDeviceToHost   );

  if (handle->cudastat==cudaSuccess && handle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
    handle->cudastat = cudaMemcpy( target.entries, handle->matrix.entries
                                 , handle->matrix.nnz * cuZcplx_size
                                 , cudaMemcpyDeviceToHost   );
}

void delete_cudaCmplxDoubleHilbertspcVect
              ( cudaCmplxDoubleHilbertspcVectHandle* handle ) {
  handle->cudastat = cudaFree(handle->vector);
}

void delete_cudaCmplxDoubleHilbertspcGenM
              ( cudaCmplxDoubleHilbertspcGenMHandle* handle ) {
  handle->cudastat = cudaFree(handle->matrix.entries);
}

void delete_cudaCmplxDoubleHilbertspcSparseM
              ( cudaCmplxDoubleHilbertspcSparseMHandle* handle ) {
  handle->cudastat = cudaFree(handle->matrix.entries);
  handle->cudastat = cudaFree(handle->matrix.row);
  handle->cudastat = cudaFree(handle->matrix.col);
  
  cusparseDestroyMatDescr(handle->matrxdescript);
  
  if(handle->inverseapphd) {
    handle->cusparsestat = cusparseDestroySolveAnalysisInfo(*handle->inverseapphd);
    delete handle->inverseapphd;
  }
}

void delete_cudaCmplxDoubleHilbertspcEigenbasisTransform
              ( cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle ) {
  if(handle->lhst_present) cudaFree(handle->lhst.entries);
  if(handle->rhst_present) cudaFree(handle->rhst.entries);
  if(handle->n_realeigenvals>0) cudaFree(handle->real_eigenvals);
  if(handle->n_cplxeigenvals>0) cudaFree(handle->complex_eigenvals);
}



cudaCmplxDoubleHilbertspcVectHandle copy_cudaCmplxDoubleHilbertspcVect
              ( cudaCmplxDoubleHilbertspcVectHandle* cpyhandle ) {
  cudaCmplxDoubleHilbertspcVectHandle handle = *cpyhandle;
  handle.cudastat = cudaMalloc( (void**)&handle.vector
                              , handle.vect_dimension * cuZcplx_size );
  if (handle.cudastat==cudaSuccess)
    handle.cublasstat = cublasZcopy( *handle.cublashandle
                                   , handle.vect_dimension
                                   , cpyhandle->vector, 1
                                   , handle.vector, 1      );
  return handle;
}

cudaCmplxDoubleHilbertspcGenMHandle copy_cudaCmplxDoubleHilbertspcGenM
              ( cudaCmplxDoubleHilbertspcGenMHandle* cpyhandle ) {
  cudaCmplxDoubleHilbertspcGenMHandle handle = *cpyhandle;
  
  unsigned arraysz = cudaCmplxDoubleHilbertspcGenM_datasize(&handle);
  
  handle.cudastat = cudaMalloc( (void**)&handle.matrix.entries
                              , arraysz );
  if (handle.cudastat==cudaSuccess)
    handle.cublasstat = cublasZcopy( *handle.cublashandle
                                   , arraysz / cuZcplx_size
                                   , cpyhandle->matrix.entries, 1
                                   , handle.matrix.entries,     1 );
  return handle;
}

cudaCmplxDoubleHilbertspcSparseMHandle copy_cudaCmplxDoubleHilbertspcSparseM
              ( cudaCmplxDoubleHilbertspcSparseMHandle* cpyhandle ) {
  cudaCmplxDoubleHilbertspcSparseMHandle handle;
  handle.cusparsehandle = cpyhandle->cusparsehandle;
  handle.cusparsestat = cpyhandle->cusparsestat;
  handle.matrix = cpyhandle->matrix;
  handle.matrxdescript = copy_sparsemat_descr_t(cpyhandle->matrxdescript);
  handle.inverseapphd = NULL;

  if ((handle.cudastat=cudaSuccess)==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.entries
                                , handle.matrix.nnz * cuZcplx_size   );
//  if (handle.cudastat==cudaSuccess) printf("Allocated space for sparse matrix entries\n");
  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.row
                                , (handle.matrix.cntvardim+1) * sizeof(int) );
//  if (handle.cudastat==cudaSuccess) printf("Allocated space for sparse matrix row ptrs\n");
  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.col
                                , handle.matrix.nnz * sizeof(int)    );
//  if (handle.cudastat==cudaSuccess) printf("Allocated space for sparse matrix column ptrs\n");

  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMemcpy( handle.matrix.entries, cpyhandle->matrix.entries
                                , handle.matrix.nnz * cuZcplx_size
                                , cudaMemcpyDeviceToDevice                );
//  if (handle.cudastat==cudaSuccess) printf("Copied sparse matrix entries\n");
  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMemcpy( handle.matrix.col, cpyhandle->matrix.col
                                , handle.matrix.nnz * sizeof(int)
                                , cudaMemcpyDeviceToDevice                );
//  if (handle.cudastat==cudaSuccess) printf("Copied sparse matrix column ptrs\n");
  if (handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMemcpy( handle.matrix.row, cpyhandle->matrix.row
                                , (handle.matrix.cntvardim+1) * sizeof(int)
                                , cudaMemcpyDeviceToDevice        );
//  if (handle.cudastat==cudaSuccess) printf("Copied sparse matrix row ptrs\n");

  return handle;
}

cudaCmplxDoubleHilbertspcGenMHandle copy_cudaCmplxDoubleHilbertspcSparseMToGenM
              ( cudaCmplxDoubleHilbertspcSparseMHandle* cpyhandle
              , const cublasHandle_t* cublashandle                    ) {
  cudaCmplxDoubleHilbertspcGenMHandle handle;
  handle.cublashandle = cublashandle;

  cusparseMatDescr_t cpymode = copy_sparsemat_descr_t(cpyhandle->matrxdescript);

  switch(cusparseGetMatType(cpymode)){
   case CUSPARSE_MATRIX_TYPE_GENERAL:
    handle.matrix.matrixtype = cuCmplxDouble_GENERIC_MATRIX;
    handle.matrix.matrixpacking = cuCmplxDouble_MATPACKING_NONE;
    break;
   case CUSPARSE_MATRIX_TYPE_HERMITIAN:
    handle.matrix.matrixtype = cuCmplxDouble_HERMITIAN_MATRIX;
    handle.matrix.matrixpacking
       = cusparseGetMatFillMode(cpyhandle->matrxdescript) == CUSPARSE_FILL_MODE_UPPER
            ? cuCmplxDouble_MATUNPACKED_DEFTRIANGUPPER
            : cuCmplxDouble_MATUNPACKED_DEFTRIANGLOWER;
    cusparseSetMatType(cpymode, CUSPARSE_MATRIX_TYPE_GENERAL);
    break;
   default:
    printf("Only general or hermitian sparse matrix types copyable to dense.");
    abort();
  }

  handle.matrix.covardim = cpyhandle->matrix.covardim;
  handle.matrix.unpacked_cntvardim = cpyhandle->matrix.cntvardim;
//  printf("Densify sparse matrix...\n");
  handle.cudastat = cudaMalloc( (void**)&handle.matrix.entries
                              , handle.matrix.covardim * handle.matrix.unpacked_cntvardim * cuZcplx_size );
//  if(handle.cudastat==cudaSuccess) printf(" allocated memory for densification of sparse matrix\n");
  if (handle.cudastat==cudaSuccess)
    cpyhandle->cusparsestat
         = cusparseZcsr2dense( *cpyhandle->cusparsehandle
                             , handle.matrix.unpacked_cntvardim
                             , handle.matrix.covardim
                             , cpymode
                             , cpyhandle->matrix.entries
                             , cpyhandle->matrix.row
                             , cpyhandle->matrix.col
                             , handle.matrix.entries
                             , handle.matrix.unpacked_cntvardim  );
  if(handle.cudastat==cudaSuccess && cpyhandle->cusparsestat==CUSPARSE_STATUS_SUCCESS) {
//    printf(" copied sparse matrix into dense representation.\n");
    handle.cublasstat=CUBLAS_STATUS_SUCCESS;
    handle.culastat = culaNoError;
  }
//    printf(" copied sparse matrix into dense representation.\n");

  /*if(handle.cudastat==cudaSuccess) {
    cuDoubleComplex* hostmem = (cuDoubleComplex*)malloc(handle.matrix.covardim * handle.matrix.covardim * cuZcplx_size);
    for(int i=0; i<4; ++i) {
      for(int j=0; j<4; ++j) {
        printf("(%f+%fi) ", cuCreal(hostmem[i+handle.matrix.covardim*j]), cuCimag(hostmem[i+handle.matrix.covardim*j]));
      }printf("\n");
    }
    printf(" check if memory is ok...\n");
    handle.cudastat=cudaMemcpy( hostmem, handle.matrix.entries
              , handle.matrix.covardim * handle.matrix.covardim * cuZcplx_size
              , cudaMemcpyDeviceToHost                                              );
    if(handle.cudastat!=cudaSuccess) printf(" (apparently not!)\n");
    for(int i=0; i<4; ++i) {
      for(int j=0; j<4; ++j) {
        printf("(%f+%fi) ", cuCreal(hostmem[i+handle.matrix.covardim*j]), cuCimag(hostmem[i+handle.matrix.covardim*j]));
      }printf("\n");
    }
    free(hostmem);
  }*/
  cusparseDestroyMatDescr(cpymode);

  return handle;
}


void axpy_cudaCmplxDoubleHilbertspcVect
              ( cuDoubleComplex alpha
              , cudaCmplxDoubleHilbertspcVectHandle* xhandle
              , cudaCmplxDoubleHilbertspcVectHandle* yhandle ){
  // assert(xhandle->vect_dimension == yhandle->vect_dimension);
  xhandle->cublasstat = yhandle->cublasstat
     = cublasZaxpy( *xhandle->cublashandle
                  , xhandle->vect_dimension
                  , &alpha
                  , xhandle->vector, 1
                  , yhandle->vector, 1       );
}

void axpy_cudaCmplxDoubleHilbertspcGenM
              ( cuDoubleComplex alpha
              , cudaCmplxDoubleHilbertspcGenMHandle* xhandle
              , cudaCmplxDoubleHilbertspcGenMHandle* yhandle ){

  assert(xhandle->matrix.matrixtype == xhandle->matrix.matrixtype);       //addition of differently
  assert(xhandle->matrix.matrixpacking == xhandle->matrix.matrixpacking);// stored matrices TODO

  xhandle->cublasstat = yhandle->cublasstat
     = cublasZaxpy( *xhandle->cublashandle
                  , cudaCmplxDoubleHilbertspcGenM_datasize(xhandle)
                  , &alpha
                  , xhandle->matrix.entries, 1
                  , yhandle->matrix.entries, 1       );
}


void gemv_cudaCmplxDoubleHilbertspcGenMToVect      // y <- α⋅A x + β y
              ( cuDoubleComplex alpha
              , cuDoubleComplex beta
              , cudaCmplxDoubleHilbertspcGenMHandle* mhandle
              , /*const*/cudaCmplxDoubleHilbertspcVectHandle* xhandle
              , cudaCmplxDoubleHilbertspcVectHandle* yhandle          ){
  assert(xhandle->vect_dimension == mhandle->matrix.covardim);
  
  switch(mhandle->matrix.matrixtype) {
   case(cuCmplxDouble_GENERIC_MATRIX):
   case(cuCmplxDouble_UNITARY_MATRIX):
    assert(yhandle->vect_dimension == mhandle->matrix.unpacked_cntvardim);
    
    mhandle->cublasstat
       = cublasZgemv( *mhandle->cublashandle
                    , CUBLAS_OP_N
                    , yhandle->vect_dimension
                    , xhandle->vect_dimension
                    , &alpha
                    , mhandle->matrix.entries
                    , yhandle->vect_dimension  //lda
                    , xhandle->vector, 1
                    , &beta
                    , yhandle->vector, 1       );
    break;
   case(cuCmplxDouble_TRIANGULAR_MATRIX):
    printf("Triangular gemv not supported. (trmv/tpmv work differently)\n");
    abort();
/*    if(yhandle->cublasstat == cudaSuccess)
      mhandle->cublasstat
         = cublasZtpmv( *mhandle->cublashandle
                      , CUBLAS_FILL_MODE_LOWER
                      , CUBLAS_OP_N
                      , yhandle->vect_dimension
                      , xhandle->vect_dimension
                      , alpha
                      , mhandle->matrix.entries
                      , yhandle->vect_dimension
                      , xhandle->vector, 1
                      , beta
                      , yhandle->vector, 1       );*/
    break;
   case(cuCmplxDouble_HERMITIAN_MATRIX):
    assert(xhandle->vect_dimension==yhandle->vect_dimension);
    assert(xhandle->vect_dimension*(yhandle->vect_dimension+1)==mhandle->matrix.packed_n_entries*2);
    assert(mhandle->matrix.matrixpacking != cuCmplxDouble_MATPACKING_TRIANG_UPPER
        && mhandle->matrix.matrixpacking != cuCmplxDouble_MATPACKING_TRIANG_LOWER );
    mhandle->cublasstat
       = cublasZhemv( *mhandle->cublashandle
                    , mhandle->matrix.matrixpacking == cuCmplxDouble_MATUNPACKED_DEFTRIANGUPPER
                         ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER
                    , yhandle->vect_dimension
                    , &alpha
                    , mhandle->matrix.entries
                    , yhandle->vect_dimension  //lda
                    , xhandle->vector, 1
                    , &beta
                    , yhandle->vector, 1       );
  }
  xhandle->cublasstat=yhandle->cublasstat
      = mhandle->cublasstat;
}

void gemv_cudaCmplxDoubleHilbertspcSparseMToVect      // y <- α⋅A x + β y
              ( cuDoubleComplex alpha
              , cuDoubleComplex beta
              , cudaCmplxDoubleHilbertspcSparseMHandle* mhandle
              , /*const*/cudaCmplxDoubleHilbertspcVectHandle* xhandle
              , cudaCmplxDoubleHilbertspcVectHandle* yhandle          ){
  // assert(xhandle->vect_dimension == mhandle->matrix.covardim);
  // assert(yhandle->vect_dimension == mhandle->matrix.cntvardim);
  cusparseMatDescr_t matrxdescript = mhandle->matrxdescript;
  if(cusparseGetMatType(matrxdescript) == CUSPARSE_MATRIX_TYPE_TRIANGULAR) {
    printf("gemv-Multiplication of sparse triangular matrix not supported.\n");
    abort();
   // cusparseSetMatType(matrxdescript, CUSPARSE_MATRIX_TYPE_GENERAL);
   }/*else if(cusparseGetMatType(matrxdescript) == CUSPARSE_MATRIX_TYPE_GENERAL){
    printf("gemv with general sparse matrix.\n");
   }else if(cusparseGetMatType(matrxdescript) == CUSPARSE_MATRIX_TYPE_HERMITIAN){
    printf("gemv with hermitian sparse matrix.\n");
  }*/

  mhandle->cusparsestat
     = cusparseZcsrmv( *mhandle->cusparsehandle
                     , CUSPARSE_OPERATION_NON_TRANSPOSE
                     , yhandle->vect_dimension
                     , xhandle->vect_dimension
                     , alpha
                     , matrxdescript
                     , mhandle->matrix.entries
                     , mhandle->matrix.row
                     , mhandle->matrix.col
                     , xhandle->vector
                     , beta
                     , yhandle->vector                  );
  if (mhandle->cusparsestat != CUSPARSE_STATUS_SUCCESS)
    xhandle->cublasstat=yhandle->cublasstat
      = CUBLAS_STATUS_EXECUTION_FAILED;
}


void invapply_cudaCmplxDoubleHilbertspcGenMToVect      // y <- A⁻¹⋅y
              ( cudaCmplxDoubleHilbertspcGenMHandle* mhandle
              , cudaCmplxDoubleHilbertspcVectHandle* yhandle          ) {
  // assert(yhandle->vect_dimension ≡ mhandle->matrix.covardim ≡ mhandle->matrix.cntvardim);
  switch(mhandle->matrix.matrixtype) {
   case(cuCmplxDouble_GENERIC_MATRIX):
//    if(mhandle->cudastat==cudaSuccess) printf("Perform matrix inverse application of generic matrix...\n");

    int* pivotarr;
    mhandle->cudastat = cudaMalloc( (void**)&pivotarr
                                 , mhandle->matrix.covardim * sizeof(int) );
//    if(mhandle->cudastat==cudaSuccess) printf(" allocated memory for pivot array for matrix inverse\n");

    cuDoubleComplex* tmpmatrix;
    if(mhandle->cudastat==cudaSuccess)
      mhandle->cudastat = cudaMalloc( (void**)&tmpmatrix
                                    , mhandle->matrix.covardim * mhandle->matrix.covardim * cuZcplx_size );
//    if(mhandle->cudastat==cudaSuccess) printf(" allocated %d*%d memory for temporary matrix for matrix inverse\n",mhandle->matrix.covardim, mhandle->matrix.covardim);

    /*if(mhandle->cudastat==cudaSuccess) {
      cuDoubleComplex* hostmem = (cuDoubleComplex*)malloc(mhandle->matrix.covardim * mhandle->matrix.covardim * cuZcplx_size);
      printf(" check if memory is ok...\n");
      mhandle->cudastat=cudaMemcpy( hostmem, mhandle->matrix.entries
                , mhandle->matrix.covardim * mhandle->matrix.covardim * cuZcplx_size
                , cudaMemcpyDeviceToHost                                              );
      if(mhandle->cudastat==cudaSuccess) printf(" (apparently yes!)\n");
      for(int i=0; i<mhandle->matrix.covardim; ++i) {
        for(int j=0; j<mhandle->matrix.covardim; ++j) {
          printf("(%2.1f+%2.1fi) ", cuCreal(hostmem[i+mhandle->matrix.covardim*j]), cuCimag(hostmem[i+mhandle->matrix.covardim*j]));
        }printf("\n");
      }
      free(hostmem);
    }*/

    if(mhandle->cudastat==cudaSuccess)
      mhandle->cudastat = cudaMemcpy( tmpmatrix, mhandle->matrix.entries
                                    , mhandle->matrix.covardim * mhandle->matrix.covardim * cuZcplx_size
                                    , cudaMemcpyDeviceToDevice              );
//    if(mhandle->cudastat==cudaSuccess) printf(" copied matrix to temporary memory\n");

    if(mhandle->cudastat==cudaSuccess)
      mhandle->culastat
        = culaDeviceZgesv( mhandle->matrix.covardim
                         , 1
                         , (culaDeviceDoubleComplex*)mhandle->matrix.entries
                         , mhandle->matrix.covardim //lda of matrix
                         , pivotarr
                         , (culaDeviceDoubleComplex*)yhandle->vector
                         , mhandle->matrix.covardim //lda of vector
                         );
/*    if(mhandle->cudastat==cudaSuccess && mhandle->culastat == culaNoError)
       printf(" performed matrix inverse application.\n");*/
    cudaFree(pivotarr);    
    cudaFree(tmpmatrix);
    break;
   case(cuCmplxDouble_TRIANGULAR_MATRIX):
    if(mhandle->matrix.matrixpacking==cuCmplxDouble_MATPACKING_NONE) {
      mhandle->cublasstat
       = cublasZtrsv( *mhandle->cublashandle
                    , CUBLAS_FILL_MODE_LOWER
                    , CUBLAS_OP_N
                    , CUBLAS_DIAG_NON_UNIT
                    , yhandle->vect_dimension
                    , mhandle->matrix.entries
                    , yhandle->vect_dimension   //lda
                    , yhandle->vector
                    , 1                        );
     }else{
      printf("packed triangle inverse not supported yet"); abort();
    }
    break;
   case(cuCmplxDouble_UNITARY_MATRIX):      // U⁻¹ = U†
    assert(mhandle->matrix.matrixpacking==cuCmplxDouble_MATPACKING_NONE);

    if(mhandle->cudastat == cudaSuccess){
      cuDoubleComplex* tmpvector;
      mhandle->cudastat
         = cudaMalloc( (void**)&tmpvector, yhandle->vect_dimension * cuZcplx_size );
      if(mhandle->cudastat == cudaSuccess)
        mhandle->cudastat
           = cudaMemcpy( tmpvector, yhandle->vector
                       , yhandle->vect_dimension * cuZcplx_size
                       , cudaMemcpyDeviceToDevice               );
      
      cuDoubleComplex one = make_cuDoubleComplex(1., 0.)
                    , zero = make_cuDoubleComplex(1., 0.);
      if(mhandle->cudastat == cudaSuccess)
        mhandle->cublasstat
           = cublasZgemv( *mhandle->cublashandle
                        , CUBLAS_OP_C    // Hermitian conjugate
                        , yhandle->vect_dimension
                        , yhandle->vect_dimension
                        , &one
                        , mhandle->matrix.entries
                        , yhandle->vect_dimension   //lda
                        , tmpvector
                        , 1
                        , &zero
                        , yhandle->vector
                        , 1                        );

      cudaFree(tmpvector);
    }
    break;
   case(cuCmplxDouble_HERMITIAN_MATRIX):
    printf("Hermitian inverse not supported yet"); abort();
    break;
  }
}


void geninvapply_cudaCmplxDoubleHilbertspcTriangSparseMToVect      // y <- A⁻¹α⋅x
              ( cuDoubleComplex alpha
              , cudaCmplxDoubleHilbertspcSparseMHandle* mhandle
              , /*const*/cudaCmplxDoubleHilbertspcVectHandle* xhandle
              , cudaCmplxDoubleHilbertspcVectHandle* yhandle          ) {
  /* assert(xhandle->vect_dimension ≡ yhandle->vect_dimension
              ≡ mhandle->matrix.covardim ≡ mhandle->matrix.cntvardim); */
  if (!mhandle->inverseapphd) {
    mhandle->inverseapphd = new cusparseSolveAnalysisInfo_t;
    mhandle->cusparsestat = cusparseCreateSolveAnalysisInfo(mhandle->inverseapphd);
    if(mhandle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
      mhandle->cusparsestat
       = cusparseZcsrsv_analysis( *mhandle->cusparsehandle
                                , CUSPARSE_OPERATION_NON_TRANSPOSE
                                , yhandle->vect_dimension
                                , mhandle->matrxdescript
                                , mhandle->matrix.entries
                                , mhandle->matrix.row
                                , mhandle->matrix.col
                                , *mhandle->inverseapphd );
  }
  if(mhandle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
      mhandle->cusparsestat
       = cusparseZcsrsv_solve( *mhandle->cusparsehandle
                             , CUSPARSE_OPERATION_NON_TRANSPOSE
                             , yhandle->vect_dimension
                             , alpha
                             , mhandle->matrxdescript
                             , mhandle->matrix.entries
                             , mhandle->matrix.row
                             , mhandle->matrix.col
                             , *mhandle->inverseapphd
                             , xhandle->vector
                             , yhandle->vector                  );
}



cudaCmplxDoubleHilbertspcGenMHandle inverted_cudaCmplxDoubleHilbertspcGenM
              ( cuDoubleComplex multiplier
              , cudaCmplxDoubleHilbertspcSparseMHandle* mhandle ) {
  printf("Direct inversion of dense matrix not yet implemented.\n");
  abort();
}


cudaCmplxDoubleHilbertspcGenMHandle inverted_cudaCmplxDoubleHilbertspcSparseM_asGenM
              ( cuDoubleComplex multiplier
              , cudaCmplxDoubleHilbertspcSparseMHandle* mhandle
              , const cublasHandle_t* cublashandle             )  {
  cudaCmplxDoubleHilbertspcGenMHandle handle;
  handle.cublashandle = cublashandle;
  handle.matrix.matrixtype = cuCmplxDouble_GENERIC_MATRIX;
  handle.matrix.matrixpacking = cuCmplxDouble_MATPACKING_NONE;
  handle.matrix.covardim = mhandle->matrix.covardim;
  handle.matrix.unpacked_cntvardim = mhandle->matrix.covardim;

  cuDoubleComplex* tmpmatrix;
  handle.cudastat = cudaMalloc( (void**)&tmpmatrix
                              , handle.matrix.covardim * handle.matrix.covardim * cuZcplx_size );
  if(handle.cudastat==cudaSuccess)
    handle.cudastat = cudaMalloc( (void**)&handle.matrix.entries
                                , handle.matrix.covardim * handle.matrix.covardim * cuZcplx_size );

  assert(cusparseGetMatType(mhandle->matrxdescript) == CUSPARSE_MATRIX_TYPE_GENERAL);
  if (handle.cudastat==cudaSuccess)
    mhandle->cusparsestat
         = cusparseZcsr2dense( *mhandle->cusparsehandle
                             , handle.matrix.covardim
                             , handle.matrix.covardim
                             , mhandle->matrxdescript
                             , mhandle->matrix.entries
                             , mhandle->matrix.row
                             , mhandle->matrix.col
                             , tmpmatrix
                             , handle.matrix.covardim  );
  if(handle.cudastat==cudaSuccess && mhandle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
    handle.cublasstat=CUBLAS_STATUS_SUCCESS;

  int* pivotarr;
  if(handle.cudastat==cudaSuccess && mhandle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
    mhandle->cudastat = cudaMalloc( (void**)&pivotarr
                                 , mhandle->matrix.covardim * sizeof(int) );

  cuDoubleComplex zero= make_cuDoubleComplex(0.,0.);
  if(handle.cudastat==cudaSuccess && mhandle->cusparsestat==CUSPARSE_STATUS_SUCCESS)
    handle.culastat
      = culaDeviceZlaset( 'A'                        //scaled identity matrix
                        , handle.matrix.covardim
                        , handle.matrix.covardim
                        , *(culaDoubleComplex*)&zero
                        , *(culaDoubleComplex*)&multiplier
                        , (culaDeviceDoubleComplex*)handle.matrix.entries
                        , handle.matrix.covardim
                        );

  if(handle.cudastat==cudaSuccess && mhandle->cusparsestat==CUSPARSE_STATUS_SUCCESS && handle.culastat==culaNoError)
    handle.culastat
      = culaDeviceZgesv( handle.matrix.covardim
                       , handle.matrix.covardim
                       , (culaDeviceDoubleComplex*)tmpmatrix
                       , mhandle->matrix.covardim //lda of matrix
                       , pivotarr
                       , (culaDeviceDoubleComplex*)handle.matrix.entries
                       , mhandle->matrix.covardim //lda of solution
                       );

  cudaFree(tmpmatrix);
  cudaFree(pivotarr);

  return handle;
}



cuDoubleComplex dotc_cudaCmplxDoubleHilbertspcVect
              ( cudaCmplxDoubleHilbertspcVectHandle* xhandle
              , cudaCmplxDoubleHilbertspcVectHandle* yhandle ){
  cuDoubleComplex result;
    xhandle->cublasstat = yhandle->cublasstat
     = cublasZdotc( *xhandle->cublashandle
                  , xhandle->vect_dimension
                  , xhandle->vector, 1
                  , yhandle->vector, 1
                  , &result                 );
  return result;
}



__global__ void phaserotation_cudaCmplxDoubleVect
              ( double alpha
              , const double* thetas
              , const cuDoubleComplex* x
              , cuDoubleComplex* res
              , int N                    ){
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if(j<N) {
 // double theta = alpha * thetas[j];
//  res[j] = cuCmul(x[j],make_cuDoubleComplex(cos(theta),sin(theta)));
    double s,c;
    sincos(alpha * thetas[j], &s, &c);
    res[j] = cuCmul(x[j],make_cuDoubleComplex(s,c));
  }
}

cudaCmplxDoubleHilbertspcVectHandle phaserotation_cudaCmplxDoubleHilbertspcVect
              ( double alpha, double* thetas
              , const cudaCmplxDoubleHilbertspcVectHandle* xhandle ){
  cudaCmplxDoubleHilbertspcVectHandle reshandle = *xhandle;
  reshandle.cudastat = cudaMalloc( (void**)&reshandle.vector
                                 , reshandle.vect_dimension * cuZcplx_size );
  if(reshandle.cudastat!=cudaSuccess) return reshandle;
  static int block_size = 256;
  int n_blocks = xhandle->vect_dimension/block_size
              + (xhandle->vect_dimension%block_size==0? 0 : 1);

  phaserotation_cudaCmplxDoubleVect<<< n_blocks, block_size >>>
                                   ( alpha, thetas
                                   , xhandle->vector
                                   , reshandle.vector
                                   , xhandle->vect_dimension );
  return reshandle;
}





cudaCmplxDoubleHilbertspcEigenbasisTransformHandle eigenbasistransform_of_cudaCmplxDoubleHilbertspcGenM
              ( cudaCmplxDoubleHilbertspcGenMHandle* handle ) {
  //assert(handle.matrix.covardim == handle.matrix.unpacked_cntvardim);

  cudaCmplxDoubleHilbertspcEigenbasisTransformHandle result;
  result.cublashandle = handle->cublashandle;
  result.lhst.matrixpacking = result.rhst.matrixpacking
     = cuCmplxDouble_MATPACKING_NONE;
  result.lhst.covardim = result.lhst.unpacked_cntvardim
     = result.rhst.covardim = result.rhst.unpacked_cntvardim
     = handle->matrix.covardim;

  result.culastat=culaNoError;

  cuDoubleComplex* opcpy;
  result.cudastat = cudaMalloc( (void**)&opcpy
                              , result.lhst.covardim * result.lhst.covardim * cuZcplx_size );
  if (result.cudastat==cudaSuccess)
    result.cublasstat = cublasZcopy( *result.cublashandle
                                   , result.lhst.covardim * result.lhst.covardim
                                   , handle->matrix.entries, 1
                                   , opcpy                 , 1 );

  if (result.cudastat==cudaSuccess && result.cublasstat==CUBLAS_STATUS_SUCCESS) {
    switch (handle->matrix.matrixtype) {
//     case cuCmplxDouble_GENERIC_MATRIX:
     case cuCmplxDouble_HERMITIAN_MATRIX:
      result.lhst_present = false;

      result.rhst_present = true;
      result.rhst.matrixtype = cuCmplxDouble_UNITARY_MATRIX;

      result.n_cplxeigenvals = 0;
      result.complex_eigenvals = NULL;
      
      result.n_realeigenvals = result.lhst.covardim;
      result.cudastat = cudaMalloc( (void**)&result.real_eigenvals
                                  , result.lhst.covardim * sizeof(double) );

      if (result.cudastat==cudaSuccess)
        result.culastat = culaDeviceZheev( 'V'
                                         , 'L'
                                         , result.lhst.covardim
                                         , (culaDeviceDoubleComplex*)opcpy
                                         , result.lhst.covardim
                                         , result.real_eigenvals );
      result.rhst.entries = opcpy;
      opcpy = NULL;

      break;
     default:
      printf("Generic-matrix eigenbasis transformation not implemented yet.");
      abort();
    }
  }

  cudaFree(opcpy);

  return result;
}


cudaCmplxDoubleHilbertspcVectHandle cudaCmplxDoubleHilbertspcVect_transformToEigenbasis
              ( const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* eigenbs
              , const cudaCmplxDoubleHilbertspcVectHandle* xhandle             ){

  cudaCmplxDoubleHilbertspcVectHandle result = undefined_cudaCmplxDoubleHilbertspcVect
           ( eigenbs->n_realeigenvals + eigenbs->n_cplxeigenvals
           , eigenbs->cublashandle                               );

  cuDoubleComplex one = make_cuDoubleComplex(1.,0.)
                , zero = make_cuDoubleComplex(0.,0.);

  if(eigenbs->rhst_present){
    switch(eigenbs->rhst.matrixtype) {
     case(cuCmplxDouble_UNITARY_MATRIX):
      result.cublasstat
         = cublasZgemv( *eigenbs->cublashandle
                      , CUBLAS_OP_C
                      , result.vect_dimension
                      , xhandle->vect_dimension
                      , &one
                      , eigenbs->rhst.entries
                      , result.vect_dimension  //lda
                      , xhandle->vector, 1
                      , &zero
                      , result.vector,   1       );
      break;
     default:
      printf("Trying to enter operator eigenbasis by unsupported transformation type\n");
      abort();
    }
   }else{
    printf("Trying to enter operator eigenbasis with no right-eigen-matrix present in transform\n");
    abort();
  }

  return result;
}

cudaCmplxDoubleHilbertspcVectHandle cudaCmplxDoubleHilbertspcVect_transformFromEigenbasis
              ( const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* eigenbs
              , const cudaCmplxDoubleHilbertspcVectHandle* xhandle             ){

  cudaCmplxDoubleHilbertspcVectHandle result = undefined_cudaCmplxDoubleHilbertspcVect
           ( eigenbs->n_realeigenvals
           , eigenbs->cublashandle    );

  cuDoubleComplex one = make_cuDoubleComplex(1.,0.)
                , zero = make_cuDoubleComplex(0.,0.);

  if(eigenbs->rhst_present){
    switch(eigenbs->rhst.matrixtype) {
     case(cuCmplxDouble_GENERIC_MATRIX):
     case(cuCmplxDouble_UNITARY_MATRIX):
      result.cublasstat
         = cublasZgemv( *eigenbs->cublashandle
                      , CUBLAS_OP_N
                      , result.vect_dimension
                      , xhandle->vect_dimension
                      , &one
                      , eigenbs->rhst.entries
                      , result.vect_dimension  //lda
                      , xhandle->vector, 1
                      , &zero
                      , result.vector,   1       );
      break;
     default:
      printf("Trying to leave operator eigenbasis by unsupported transformation type\n");
      abort();
    }
   }else{
    printf("Trying to leave operator eigenbasis with no right-eigen-matrix present in transform\n");
    abort();
  }
  return result;
}




#if 0
  printf("Perform phase-rotations on CUDA with %d blocks of size %d (%d elements total)\n"
                                  ,         n_blocks    , block_size,xhandle->vect_dimension);
  double view[1024];
  cublasGetVector( xhandle->vect_dimension, cuZcplx_size
                 , xhandle->vector, 1, view, 1           );
  printf("some of the inputs:\n(real)    %8.4f,%8.4f,%8.4f", view[32], view[64], view[128]);
  printf(                 "\n(imag)    %8.4f,%8.4f,%8.4f\n", view[33], view[65], view[129]);
  cublasGetVector( xhandle->vect_dimension, sizeof(double)
                 , thetas, 1, view, 1           );
  printf("some of the angles:\n(as given)%8.4f,%8.4f,%8.4f", view[32], view[64], view[128]);
  printf(                   "\n(scaled)  %8.4f,%8.4f,%8.4f\n", alpha*view[32], alpha*view[64], alpha*view[128]);

  cublasGetVector( xhandle->vect_dimension, cuZcplx_size
                 , reshandle.vector, 1, view, 1           );
  printf("some of the results:\n(real)    %8.4f,%8.4f,%8.4f", view[32], view[64], view[128]);
  printf(                 "\n(imag)    %8.4f,%8.4f,%8.4f\n", view[33], view[65], view[129]);
#endif
