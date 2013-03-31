/*
 * =====================================================================================
 *
 *       Filename:  matrixMultiplicationWithoutSharedMemory.c
 *
 *    Description:  based on CUDA C programming    
 *    			    This code multiplies two matrices A and B and gives matrix C as 
 *    			    the result, This code runs on GPU when all threads have no shared memory 
 *    				This is matrix multiplication on host
 *    				or I can say its host code  
 *    				Matrix dimensions are assumed to be multiples of BLOCK_SIZE 
 *
 *     USAGE: ./matrixMultiplicationWithoutSharedMemory a1 a2 b2 
 *     Example ./mulNoShare 4 2 5
 *
 *        Version:  1.0
 *        Created:  04/01/2013 00:40:06
 *       Revision:  none
 *       Compiler:  gcc, nvcc 
 		 Compiling: nvcc -o matrixMultiplicationWithoutSharedMemory -g -G matrixMultiplicationWithoutSharedMemory.cu 
 *
 *         Author:  ABDIMUNA (), abdimuna1@gmail.com
 *   Organization:   
         OPINIONS:  "Contributions to this code is highly encouraged!, you can contact me thro
		 			 my email, or facebook, abdimuna1" 

		DISCLAIMER:  use this code under your own modifications, since I'm still working on it, 
					 it seem the results are not what I expected, thanks 

 *
 * =====================================================================================
 */


#include "matrixMultiplicationWithoutSharedMemory.h" 
#include <cuda.h> 
#include <string.h> 



void MatMul(const Matrix A, const Matrix B, Matrix C)
{ 
	// Load A and B to device memory 
	
	Matrix d_A; 
	d_A.width = A.width; 
	d_A.height = A.height; 

	size_t size = A.width * A.height *sizeof(float); 
	cudaError_t err = cudaMalloc(&d_A.elements, size); 
	printf("CUDA malloc A:%s\n", cudaGetErrorString(err)); 
	err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
	printf("Copy A to device: %s\n", cudaGetErrorString(err));  
	

	// Loading Matrix B to device Memory 
	//
	
	Matrix d_B; 
	d_B.width  = B.width; 
	d_B.height = B.height; 
	size = B.width * B.height *sizeof(float); 
	err = cudaMalloc(&d_B.elements, size); 

	printf("CUDA malloc B: %s\n", cudaGetErrorString(err)); 
	err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice); 
	printf("Copy B to device: %s\n", cudaGetErrorString(err)); 

	// Allocate C in device memory 
	Matrix d_C; 
	d_C.width = C.width; 
	d_C.height = C.height; 
	size = C.width * C.height *sizeof(float); 
	err = cudaMalloc(&d_C.elements, size); 
	printf("CUDA malloc C: %s\n", cudaGetErrorString(err)); 

	// Invoke kernel 
	//
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y -1) / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
	err = cudaThreadSynchronize(); 
	printf("Run kernel: %s\n", cudaGetErrorString(err)); 

	// Read C from device memory
	
	err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
	printf("Copy C off the GPU: %s\n", cudaGetErrorString(err)); 

	// Free device memory 
	cudaFree(d_A.elements); 
	cudaFree(d_B.elements); 
	// cudaFree(d_C.elements); 
	
}

// Matrix multiplication kernel called by MatMul(): 
//
  __global__  void MatMulKernel(Matrix A, Matrix B, Matrix C) 
{ 
	// Each thread computes one element of C 
	// by accumulating results into Cvalue

	 float Cvalue = 0.0; 
	 int row = blockIdx.y + blockDim.y + threadIdx.y;  // vertically are rows 
	 int col = blockIdx.x + blockDim.x + threadIdx.x; // horrizontally are columns 
	 if(row > A.height || col > B.width) return; // if there are more rows or cols just return

	 for(int i= 0; i<A.width; ++i) 
	  Cvalue += (A.elements[row * A.width + i]) * (B.elements[i*B.width + col]); 
	 C.elements[row *C.width + col] = Cvalue; 

} 

// Usage mulNoShare a1 a2 b2 

int main(int argc, char **argv)
{  
	Matrix A, B, C;  
	int a1, a2, b1, b2; 

	// Read some values from commandLine 
	a1 = atoi(argv[1]); /*  Height of A i.e total rows of matrix A  */ 
	a2 = atoi(argv[2]); /*  Width of A i.e total number of columns of Matrix A  */ 
	b1 = a2; /*  Height of B i.e Number of rows of B */ 
	b2 = atoi(argv[3]); /*  Width of B, i.e Number of columns of matrix B */ 

	A.height = a1; 
	A.width = a2; 
	A.elements = (float*)malloc(A.width * A.height * sizeof(float));  

	B.height = b1; 
	B.width = b2; 
	B.elements = (float *)malloc(B.width * B.height *sizeof(float)); 

	C.height = A.height; 
	C.width = B.width; 
	C.elements = (float *)malloc(C.width *C.height *sizeof(float)); 

	for(int i = 0; i< A.height; i++)
			for(int j =0; j<A.width; j++)
			A.elements[i*A.width +j] = (float)(arc4random() % 3); 

	
	for(int i = 0; i< B.height; i++ )
			for(int j =0; j<B.width; j++)
			B.elements[i*B.width +j] = (float)(arc4random() % 2); 

	MatMul(A, B, C); 

	// printing up to a 10x10 portion of the three matrices 
	//
     // printing ---matrix A---- 
	printf("------------Matrix_A---------\n");
	for(int i = 0; i< min(10, A.height); i++)
	{ 
		for(int j = 0; j< min(10, A.width); j++)
			printf("%.3f ", A.elements[i*A.width +j]); 
		printf("\n"); 
	} 

	printf("\n");  


	// printing matirx ----B----- 
	printf("------------Matrix_B---------\n");
	for(int i = 0; i< min(10, B.height); i++)
	{ 
		for(int j = 0; j< min(10, B.width); j++)
			printf("%.3f ", B.elements[i*B.width +j]); 
		printf("\n"); 
	} 

	printf("\n"); 
	
	// printing matirx ----C----- 
	printf("------------Matrix_C---------\n");
	for(int i = 0; i< min(10, C.height); i++)
	{ 
		for(int j = 0; j< min(10, C.width); j++)
			printf("%.3f ", B.elements[i*C.width +j]); 
		printf("\n"); 
	} 

	printf("\n"); 
	
	
	return 0; 
} 









