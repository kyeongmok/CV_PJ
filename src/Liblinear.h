/* Dll API function definitions */

// #ifdef LIBLINEAR_EXPORTS
// #define LIBLINEAR_API __declspec(dllexport)
// #else
// #define LIBLINEAR_API __declspec(dllimport)
// #endif

#ifndef __LIBLINEAR_H
#define __LIBLINEAR_H
#endif


/*LIBLINEAR_API */void L2RegularL2LossSVRDual(double* pInputMatrix, int inputRow, int inputCol,
													double* pTargetMatrix, int targetRow, int targetCol,
													double* pResult, int resRow, int resCol);

