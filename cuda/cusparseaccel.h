#ifndef GENERAL_CUSPARSE_HELPER_FUNCTIONS_FOR_USE_IN_HILBERTSPACES
#define GENERAL_CUSPARSE_HELPER_FUNCTIONS_FOR_USE_IN_HILBERTSPACES

#include "cusparse.h"
#include<stdio.h>

void print_cusparse_status_message(const cusparseStatus_t& stat);

#endif