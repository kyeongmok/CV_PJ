#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <vector>
#include "linear.h"

#ifndef __LIBLINEAR_H
#include "Liblinear.h"
#endif

template <typename T>
void Transpose(const T* pInputMatrix, T* pOutputMatrix, int inputRow, int inputCol);

/*LIBLINEAR_API*/ void L2RegularL2LossSVRDual(double* pInputMatrix, int inputRow, int inputCol,
													double* pTargetMatrix, int targetRow, int targetCol,
													double* pResult, int resRow, int resCol)
{
/*  Solve Linear Regression Problem abstracted from
 *	I * W = T
 *
 *	where,
 *	i : # of input samples
 *	f : size of feature(input) vector
 *	d : size of regression target vector
 *
 *	I : i x f	input matrix
 *	W : f x d	output regressor (mapping func)
 *	T : i x d	regression target matrix
 *	
 */

	assert(inputRow == targetRow && inputCol == resRow && targetCol == resCol);

	//problem data setting
	struct problem prob12;
	prob12.l = inputRow;
	prob12.n = inputCol;
	prob12.bias = -1;
	prob12.y = new double[targetRow];
	prob12.x = new feature_node*[inputRow];
	for (int row = 0; row < inputRow; row++)
	{
		std::vector<std::pair<int, double> > inputRowArray;	// (index, value)
		for (int col = 0; col < inputCol; col++)
		{
			int val = pInputMatrix[row * inputCol + col];
			if (val != 0)
				inputRowArray.push_back(std::make_pair(col+1, val));
		}
		prob12.x[row] = new feature_node[inputRowArray.size() + 2];
		//prob12.x[row] = new feature_node[inputRowArray.size() + 1];

		for (int i = 0; i < inputRowArray.size(); i++)
		{
			prob12.x[row][i].index = inputRowArray[i].first;
			prob12.x[row][i].value = inputRowArray[i].second;
		}
		prob12.x[row][inputRowArray.size()].index = prob12.n;
		prob12.x[row][inputRowArray.size()].value = prob12.bias;
		prob12.x[row][inputRowArray.size() + 1].index = -1;
		//prob12.x[row][inputRowArray.size()].index = -1;
	}

	//parameter setting
	struct parameter param12;
	param12.solver_type = L2R_L2LOSS_SVR_DUAL;
	param12.eps = 0.1;
	param12.C = 1/(double)prob12.l;
	param12.nr_weight = 0;
	param12.weight_label = NULL;
	param12.weight = NULL;
	param12.p = 0;

	//transpose the target matrix for memory cache hit
	double* pTargetTransposed = new double[targetRow * targetCol];
	Transpose<double>(pTargetMatrix, pTargetTransposed, targetRow, targetCol);
	int tarTransRow = targetCol;
	int tarTransCol = targetRow;

	//regression for all target dimensions
	double* pTmpResult = new double[resRow * resCol];
	for (int i = 0; i < resCol; i++)
	{
		memcpy(prob12.y, pTargetTransposed + i*tarTransCol, tarTransCol * sizeof(double));
		struct model* pModel12 = train(&prob12, &param12);
		memcpy(pTmpResult + i*resRow, pModel12->w, prob12.n * sizeof(double));

		/* free the model */
		if (pModel12->param.weight_label)
			free(pModel12->param.weight_label);
		if (pModel12->param.weight)
			free(pModel12->param.weight);
		if (pModel12->label)
			free(pModel12->label);
		free(pModel12->w);
	}

	Transpose<double>(pTmpResult, pResult, resCol, resRow);

	for (int i = 0; i < inputRow; i++) 
		delete[] prob12.x[i];
	delete[] prob12.x;
	delete[] pTargetTransposed;
	delete[] prob12.y;
	delete[] pTmpResult;
	return;
}

template <typename T>
void Transpose(const T* pInputMatrix, T* pOutputMatrix, int inputRow, int inputCol)
{
	int numIter = inputRow * inputCol;
	for (int i = 0; i < numIter; i++)
	{
		int row = i % inputRow;
		int col = i / inputRow;
		pOutputMatrix[i] = pInputMatrix[row * inputCol + col];
	}
	return;
}
