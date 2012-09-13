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


#ifndef CUDA_ACCELLERATION_FOR_COMPLEXDOUBLE_HILBERTSPACES
#define CUDA_ACCELLERATION_FOR_COMPLEXDOUBLE_HILBERTSPACES

//#include <math.h>
//#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include "cuComplex.h"
#include <cula.h>
#include <cula_lapack.h>


//-------------- vector / matrix handle types -----------------//

typedef struct {
  const cublasHandle_t* cublashandle;
  int vect_dimension;
  cuDoubleComplex* vector;
  cudaError_t cudastat;
  cublasStatus_t cublasstat;
}cudaCmplxDoubleHilbertspcVectHandle;


enum cuCmplxDoubleGenMatType {
    cuCmplxDouble_GENERIC_MATRIX
  , cuCmplxDouble_TRIANGULAR_MATRIX
  , cuCmplxDouble_HERMITIAN_MATRIX
  , cuCmplxDouble_UNITARY_MATRIX
};
enum cuCmplxDoubleGenMatPacking {
    cuCmplxDouble_MATPACKING_NONE
  , cuCmplxDouble_MATUNPACKED_DEFTRIANGUPPER
  , cuCmplxDouble_MATUNPACKED_DEFTRIANGLOWER
  , cuCmplxDouble_MATPACKING_TRIANG_UPPER
  , cuCmplxDouble_MATPACKING_TRIANG_LOWER
};
typedef struct {
  cuCmplxDoubleGenMatType matrixtype;
  cuCmplxDoubleGenMatPacking matrixpacking;
  unsigned covardim;
  union{
    unsigned unpacked_cntvardim;  //when matrix symmetric, cntvardim==covardim
    unsigned packed_n_entries;   // and nentries == covardim * (covardim+1) / 2
  };
  cuDoubleComplex* entries;
}cuCmplxDoubleGenMatEntries;
typedef/*struct {                    //proper const treatment in the higher C++ classes
  unsigned covardim;unsigned cntvardim; ...
}*/cuCmplxDoubleGenMatEntries
const_cuCmplxDoubleGenMatEntries;
typedef struct {
  const cublasHandle_t* cublashandle;
  cuCmplxDoubleGenMatEntries matrix;
  cudaError_t cudastat;
  cublasStatus_t cublasstat;
  culaStatus culastat;
}cudaCmplxDoubleHilbertspcGenMHandle;



typedef struct {
  int* row; unsigned covardim;  //may be in either csr format or coo on host or device
  int* col; unsigned cntvardim;
  cuDoubleComplex* entries; unsigned nnz;
}cuCmplxDoubleSparseMatEntries;
typedef/*struct {const int* row; unsigned covardim;const int* col; unsigned cntvardim;const cuDoubleComplex* entries; unsigned nnz;
}*/cuCmplxDoubleSparseMatEntries
const_cuCmplxDoubleSparseMatEntries;

typedef struct {
  const cusparseHandle_t* cusparsehandle;
  cusparseMatDescr_t matrxdescript;
  cuCmplxDoubleSparseMatEntries matrix;   //always in csr format
  cudaError_t cudastat;
  cusparseStatus_t cusparsestat;
  cusparseSolveAnalysisInfo_t* inverseapphd;
}cudaCmplxDoubleHilbertspcSparseMHandle;


typedef struct {
  const cublasHandle_t* cublashandle;

  const_cuCmplxDoubleGenMatEntries lhst;
  bool lhst_present;
  const_cuCmplxDoubleGenMatEntries rhst;
  bool rhst_present;

  double* real_eigenvals;
  int n_realeigenvals;
  cuDoubleComplex* complex_eigenvals;
  int n_cplxeigenvals;

  cudaError_t cudastat;
  cublasStatus_t cublasstat;
  culaStatus culastat;
}cudaCmplxDoubleHilbertspcEigenbasisTransformHandle;



//-------------- error/status management -----------------//


int
  cudaCmplxDoubleHilbertspcVect_bad
  ( const cudaCmplxDoubleHilbertspcVectHandle* handle );
void
  cudaCmplxDoubleHilbertspcVect_printerrmsg
  ( const cudaCmplxDoubleHilbertspcVectHandle* handle );
void
  cudaCmplxDoubleHilbertspcVect_resetStatus
  ( cudaCmplxDoubleHilbertspcVectHandle* handle );

int
  cudaCmplxDoubleHilbertspcGenM_bad
  ( const cudaCmplxDoubleHilbertspcGenMHandle* handle );
void
  cudaCmplxDoubleHilbertspcGenM_printerrmsg
  ( const cudaCmplxDoubleHilbertspcGenMHandle* handle );
void
  cudaCmplxDoubleHilbertspcGenM_resetStatus
  ( cudaCmplxDoubleHilbertspcGenMHandle* handle );

int
  cudaCmplxDoubleHilbertspcSparseM_bad
  ( const cudaCmplxDoubleHilbertspcSparseMHandle* handle );
void
  cudaCmplxDoubleHilbertspcSparseM_printerrmsg
  ( const cudaCmplxDoubleHilbertspcSparseMHandle* handle );
void
  cudaCmplxDoubleHilbertspcSparseM_resetStatus
  ( cudaCmplxDoubleHilbertspcSparseMHandle* handle );

int
  cudaCmplxDoubleHilbertspcEigenbasisTransform_bad
  (const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle );
void
  cudaCmplxDoubleHilbertspcEigenbasisTransform_printerrmsg
  (const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle);
void
  cudaCmplxDoubleHilbertspcEigenbasisTransform_resetStatus
  (cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle);



//---------------- host/default constructors -------------------//

cudaCmplxDoubleHilbertspcVectHandle
  unassigned_cudaCmplxDoubleHilbertspcVectHandle
  ( const cublasHandle_t* cublashandle          );

cudaCmplxDoubleHilbertspcGenMHandle
  unassigned_cudaCmplxDoubleHilbertspcGenMHandle
  ( const cublasHandle_t* cublashandle          );

cudaCmplxDoubleHilbertspcSparseMHandle
  unassigned_cudaCmplxDoubleHilbertspcSparseMHandle
  ( const cusparseHandle_t* cusparsehandle          );


cudaCmplxDoubleHilbertspcVectHandle
  undefined_cudaCmplxDoubleHilbertspcVect
  ( unsigned dimension
  , const cublasHandle_t* cublashandle );

cudaCmplxDoubleHilbertspcGenMHandle
  undefined_cudaCmplxDoubleHilbertspcGenM
  ( cuCmplxDoubleGenMatType mtype
  , cuCmplxDoubleGenMatPacking mtpacking
  , unsigned covardim
  , unsigned cntvardim
  , const cublasHandle_t* cublashandle );


cudaCmplxDoubleHilbertspcVectHandle
  new_cudaCmplxDoubleHilbertspcVect
  ( const cuDoubleComplex* source     //on host
  , unsigned dimension
  , const cublasHandle_t* cublashandle     );

cudaCmplxDoubleHilbertspcGenMHandle
  new_cudaCmplxDoubleHilbertspcGenM
  ( const_cuCmplxDoubleGenMatEntries source   //on host
  , const cublasHandle_t* cublashandle );

cudaCmplxDoubleHilbertspcSparseMHandle
  new_cudaCmplxDoubleHilbertspcSparseM
  ( const_cuCmplxDoubleSparseMatEntries source   //on host, in coo format
  , const cusparseHandle_t* cusparsehandle );



//--------------------- data retrieval -------------------//

void
  get_cudaCmplxDoubleHilbertspcVect
  ( cuDoubleComplex* target
  , cudaCmplxDoubleHilbertspcVectHandle* handle );

void
  get_cudaCmplxDoubleHilbertspcGenM
  ( cuCmplxDoubleGenMatEntries target
  , cudaCmplxDoubleHilbertspcGenMHandle* handle );

void
  get_cudaCmplxDoubleHilbertspcSparseM
  ( cuCmplxDoubleSparseMatEntries target
  , cudaCmplxDoubleHilbertspcSparseMHandle* handle );


//------------------------ destructors ---------------------//

void
  delete_cudaCmplxDoubleHilbertspcVect
  ( cudaCmplxDoubleHilbertspcVectHandle* handle );

void
  delete_cudaCmplxDoubleHilbertspcGenM
  ( cudaCmplxDoubleHilbertspcGenMHandle* handle );

void
  delete_cudaCmplxDoubleHilbertspcSparseM
  ( cudaCmplxDoubleHilbertspcSparseMHandle* handle );

void
  delete_cudaCmplxDoubleHilbertspcEigenbasisTransform
  ( cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* handle );

//------------------- copy constructors ----------------------//

cudaCmplxDoubleHilbertspcVectHandle
  copy_cudaCmplxDoubleHilbertspcVect
  ( cudaCmplxDoubleHilbertspcVectHandle* handle );

cudaCmplxDoubleHilbertspcGenMHandle
  copy_cudaCmplxDoubleHilbertspcGenM
  ( cudaCmplxDoubleHilbertspcGenMHandle* handle );

cudaCmplxDoubleHilbertspcSparseMHandle
  copy_cudaCmplxDoubleHilbertspcSparseM
  ( cudaCmplxDoubleHilbertspcSparseMHandle* handle );

cudaCmplxDoubleHilbertspcGenMHandle
  copy_cudaCmplxDoubleHilbertspcSparseMToGenM
  ( cudaCmplxDoubleHilbertspcSparseMHandle* handle
  , const cublasHandle_t* cublashandle             );




//------------------- level-1 linear algebra --------------------//

void
  axpy_cudaCmplxDoubleHilbertspcVect         // y += α x
  ( cuDoubleComplex alpha
  , cudaCmplxDoubleHilbertspcVectHandle* xhandle    //const as far as the vector is concerned
  , cudaCmplxDoubleHilbertspcVectHandle* yhandle );

void
  axpy_cudaCmplxDoubleHilbertspcGenM         // y += α x
  ( cuDoubleComplex alpha
  , /*const*/cudaCmplxDoubleHilbertspcGenMHandle* xhandle
  , cudaCmplxDoubleHilbertspcGenMHandle* yhandle );


//------------------- level-2 linear algebra --------------------//

void
  gemv_cudaCmplxDoubleHilbertspcGenMToVect      // y <- α⋅A x + β y
  ( cuDoubleComplex alpha
  , cuDoubleComplex beta
  , cudaCmplxDoubleHilbertspcGenMHandle* mhandle
  , /*const*/cudaCmplxDoubleHilbertspcVectHandle* xhandle
  , cudaCmplxDoubleHilbertspcVectHandle* yhandle          );

void
  gemv_cudaCmplxDoubleHilbertspcSparseMToVect      // y <- α⋅A x + β y
  ( cuDoubleComplex alpha
  , cuDoubleComplex beta
  , cudaCmplxDoubleHilbertspcSparseMHandle* mhandle
  , /*const*/cudaCmplxDoubleHilbertspcVectHandle* xhandle
  , cudaCmplxDoubleHilbertspcVectHandle* yhandle          );

void//simple inverse application of a matrix mapping to a vector,
   // with no lasting side-effects to the matrix. Extremely
  //  inefficient except in the case of a triangular matrix.
  invapply_cudaCmplxDoubleHilbertspcGenMToVect      // y <- A⁻¹y
  ( cudaCmplxDoubleHilbertspcGenMHandle* mhandle
  , cudaCmplxDoubleHilbertspcVectHandle* yhandle          );

void
  geninvapply_cudaCmplxDoubleHilbertspcTriangSparseMToVect      // y <- A⁻¹α⋅x
  ( cuDoubleComplex alpha
  , cudaCmplxDoubleHilbertspcSparseMHandle* mhandle
  , /*const*/cudaCmplxDoubleHilbertspcVectHandle* xhandle
  , cudaCmplxDoubleHilbertspcVectHandle* yhandle          );



//--------------------- inversions -----------------------//

cudaCmplxDoubleHilbertspcGenMHandle
  inverted_cudaCmplxDoubleHilbertspcGenM
  ( cuDoubleComplex multiplier
  , cudaCmplxDoubleHilbertspcGenMHandle* handle );

cudaCmplxDoubleHilbertspcGenMHandle
  inverted_cudaCmplxDoubleHilbertspcSparseM_asGenM
  ( cuDoubleComplex multiplier
  , cudaCmplxDoubleHilbertspcSparseMHandle* handle
  , const cublasHandle_t* cublashandle             );


//-------------- misc linear/bilinear operations ------------//

cuDoubleComplex
  dotc_cudaCmplxDoubleHilbertspcVect
  ( cudaCmplxDoubleHilbertspcVectHandle* xhandle   //const vector
  , cudaCmplxDoubleHilbertspcVectHandle* yhandle   //const vector
  );

cudaCmplxDoubleHilbertspcVectHandle     // xⱼ ⋅ exp(i ⋅ α ⋅ ϑⱼ)
  phaserotation_cudaCmplxDoubleHilbertspcVect
  ( double alpha
  , double* thetas   //on device
  , const cudaCmplxDoubleHilbertspcVectHandle* xhandle
  );


//------------------- eigenvalue problems -------------------//

cudaCmplxDoubleHilbertspcEigenbasisTransformHandle
  eigenbasistransform_of_cudaCmplxDoubleHilbertspcGenM
  ( cudaCmplxDoubleHilbertspcGenMHandle* handle );

cudaCmplxDoubleHilbertspcEigenbasisTransformHandle
  eigenbasistransform_of_cudaCmplxDoubleHilbertspcSparseM
  ( cudaCmplxDoubleHilbertspcSparseMHandle* handle
  , const cublasHandle_t* cublashandle             );

cudaCmplxDoubleHilbertspcVectHandle
  cudaCmplxDoubleHilbertspcVect_transformToEigenbasis
  ( const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* eigenbasis
  , const cudaCmplxDoubleHilbertspcVectHandle* xhandle             );

cudaCmplxDoubleHilbertspcVectHandle
  cudaCmplxDoubleHilbertspcVect_transformFromEigenbasis
  ( const cudaCmplxDoubleHilbertspcEigenbasisTransformHandle* eigenbasis
  , const cudaCmplxDoubleHilbertspcVectHandle* xhandle             );


#endif