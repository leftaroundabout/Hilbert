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

#include "cusparse.h"
#include<stdio.h>


void print_cusparse_status_message(const cusparseStatus_t& stat){
  switch (stat) {
   case CUSPARSE_STATUS_SUCCESS                  : printf("CUSPARSE_STATUS_SUCCESS"                  ); break;
   case CUSPARSE_STATUS_NOT_INITIALIZED          : printf("CUSPARSE_STATUS_NOT_INITIALIZED"          ); break;
   case CUSPARSE_STATUS_ALLOC_FAILED             : printf("CUSPARSE_STATUS_ALLOC_FAILED"             ); break;
   case CUSPARSE_STATUS_INVALID_VALUE            : printf("CUSPARSE_STATUS_INVALID_VALUE"            ); break;
   case CUSPARSE_STATUS_ARCH_MISMATCH            : printf("CUSPARSE_STATUS_ARCH_MISMATCH"            ); break;
   case CUSPARSE_STATUS_MAPPING_ERROR            : printf("CUSPARSE_STATUS_MAPPING_ERROR"            ); break;
   case CUSPARSE_STATUS_EXECUTION_FAILED         : printf("CUSPARSE_STATUS_EXECUTION_FAILED"         ); break;
   case CUSPARSE_STATUS_INTERNAL_ERROR           : printf("CUSPARSE_STATUS_INTERNAL_ERROR"           ); break;
   case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: printf("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"); break;
  }
}
