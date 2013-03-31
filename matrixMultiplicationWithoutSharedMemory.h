/*
 * =====================================================================================
 *
 *       Filename:  matrixMultiplicationWithoutSharedMemory.h
 *
 *    Description:  its based cuda programming Guide 
 *
 *        Version:  1.0
 *        Created:  04/01/2013 00:30:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  ABDIMUNA (), abdimuna1@gmail.com
 *   Organization:  
 *
 * =====================================================================================
*/ 


#include <stdio.h> 


//Matrices are stored in row-major order: 
//meaning that the matrix are stored as one-dimensional array wit first row followed 
//by second rown and soon. 
//
//
// M(row, col) = *(M.elements +row*M.width + col) 
//

typedef struct{ 
	
	   int width; 
	   int height; 
	   float *elements; 
}Matrix; 


// Thread block size 
//

#define BLOCK_SIZE 16 

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);  

void MatMul(const Matrix, const Matrix, Matrix); 


