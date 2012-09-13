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


#include "cublas_v2.h"
#include<stdio.h>

void print_cublas_status_message(const cublasStatus_t& stat) {
  switch (stat) {
   case CUBLAS_STATUS_SUCCESS          : printf("CUBLAS_STATUS_SUCCESS"          ); break;
   case CUBLAS_STATUS_NOT_INITIALIZED  : printf("CUBLAS_STATUS_NOT_INITIALIZED"  ); break;
   case CUBLAS_STATUS_ALLOC_FAILED     : printf("CUBLAS_STATUS_ALLOC_FAILED"     ); break;
   case CUBLAS_STATUS_INVALID_VALUE    : printf("CUBLAS_STATUS_INVALID_VALUE"    ); break;
   case CUBLAS_STATUS_ARCH_MISMATCH    : printf("CUBLAS_STATUS_ARCH_MISMATCH"    ); break;
   case CUBLAS_STATUS_MAPPING_ERROR    : printf("CUBLAS_STATUS_MAPPING_ERROR"    ); break;
   case CUBLAS_STATUS_EXECUTION_FAILED : printf("CUBLAS_STATUS_EXECUTION_FAILED" ); break;
   case CUBLAS_STATUS_INTERNAL_ERROR   : printf("CUBLAS_STATUS_INTERNAL_ERROR"   ); break;
  }
}
