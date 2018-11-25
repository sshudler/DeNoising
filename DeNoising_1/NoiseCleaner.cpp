// Copyright (c) 2018 Sergei Shudler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <math.h>
#include <float.h>
#include "Utils.h"
#include "NoiseCleaner.h"

// Windows headers are only needed for accurate profiling of the CPU-based testing routines
#if defined(_WIN32) || defined(_WIN64) || defined(_WINDOWS)
#include <windows.h>
#endif

#define NUM_BANKS		16
#define TILE_SIZE		16
#define BLOCK_ROWS		8
#define INV_SQRT_2      0.70710678118654752440f
#define SQRT_2			1.41421356237309504880f

#define TEST_SIGNAL_FILE_1		"signal_2_14.dat"
#define TEST_REGRESS_FILE_1		"regression_2_14.gold.dat"

#define TEST_SIGNAL_FILE_2		"signal.dat"
#define TEST_REGRESS_FILE_2		"regression.gold.dat"

#define TEST_SIGNAL_FILE_3		"signal_small.dat"
#define TEST_REGRESS_FILE_3		"regression_small.gold.dat"



char* CNoiseCleaner::KERNEL_NAMES[NUM_KERNELS] = 
	{"FWT_kernel", "IWT_kernel", "Mat_Transpose_kernel", "Mat_HT_Threshold_kernel", "Mat_ST_Threshold_kernel"};

//-----------------------------------------------------------------------------------------
CNoiseCleaner::CNoiseCleaner() : 
m_oclEnv("HWT_kernels.cl", NUM_KERNELS, KERNEL_NAMES)
{
}
//-----------------------------------------------------------------------------------------
CNoiseCleaner::~CNoiseCleaner()
{
}
//-----------------------------------------------------------------------------------------
int CNoiseCleaner::CleanNoise(unsigned char *in, unsigned char *out, int width, int height, float thresh, bool isSoftThresh)
{
	unsigned int numLevelsWidth = 0;
	unsigned int numLevelsHeight = 0;
	if (!CNoiseCleaner::GetNumLevels(width, numLevelsWidth))
		return false;	// The buffer length is not a power of two
	if (!CNoiseCleaner::GetNumLevels(height, numLevelsHeight))
		return false;	// The buffer length is not a power of two


	// ------------------------------------------
	// Convert given buffer to a matrix of floats
	// ------------------------------------------
	unsigned int numPixels = width*height;
	float* pInFloatsMatrix = new float[numPixels];
	for (unsigned int i = 0; i < numPixels; i++)
		pInFloatsMatrix[i] = (float)in[i] / 255.f;
	
	cl_int                  clErr;
	cl_mem					gInBuff;
	cl_mem					gOutBuff;
	cl_mem					gPartialBuff;
	cl_ulong				kernelTime = 0;
	cl_ulong				totalKernelTime = 0;

	// -----------------------------------------------------------
	// Allocate device buffers and copy the whole matrix to device
	// -----------------------------------------------------------
	unsigned int gBuffSize = numPixels * sizeof(float);
	gInBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_READ_WRITE, gBuffSize, NULL, NULL);
	gOutBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_READ_WRITE, gBuffSize, NULL, NULL);
	gPartialBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_WRITE_ONLY, width*sizeof(float), NULL, NULL);

	clErr = clEnqueueWriteBuffer(m_oclEnv.m_cmdQ, gInBuff, CL_TRUE, 0, gBuffSize, pInFloatsMatrix, 0, NULL, NULL);
	OpenCLEnv::CheckForError(clErr, "writing input buffer data to device");


	// -------------------------------------------------------------------------------------
	// Invoke ForwardHaarTransformGPU on all the rows in the matrix simultaneously (which works on 1D buffers)
	// -------------------------------------------------------------------------------------
	bool bResult = true;
	bResult = bResult && ForwardHaarTransformGPU(gInBuff, gOutBuff, gPartialBuff, height, numLevelsWidth, width, 0, kernelTime);
	totalKernelTime += kernelTime;
	if (!bResult)
		return 1;
	
	OpenCLEnv::PrintProfilingInfo(totalKernelTime, "Forward transform on rows");


	// ---------------------------------------------------------------------------------------
	// Transpose the matrix by invoking a kernel which will transpose the matrix on the device
	// w/o bringing it back to the host memory
	// ---------------------------------------------------------------------------------------
	bResult = TransposeMatrixGPU(gOutBuff, gInBuff, width, height, totalKernelTime);
	if (!bResult)
		return 1;

	OpenCLEnv::PrintProfilingInfo(totalKernelTime, "Matrix transpose");
		
	// -----------------------------------------------------------------------------------------------------
	// For all of the rows in the transposed matrix (i.e. column in the original one) invoke ForwardHaarTransformGPU
	// -----------------------------------------------------------------------------------------------------
	bResult = true;
	totalKernelTime = 0;
	bResult = bResult && ForwardHaarTransformGPU(gInBuff, gOutBuff, gPartialBuff, width, numLevelsHeight, height, 0, kernelTime);
	totalKernelTime += kernelTime;
	if (!bResult)
		return 1;

	OpenCLEnv::PrintProfilingInfo(totalKernelTime, "Forward transform on columns");


	// -----------------------------------------------------------------
	// Apply threshold on the results of the Forward Haar Transform
	// -----------------------------------------------------------------
	bResult = MatrixThreshGPU(gOutBuff, gInBuff, numPixels, thresh, totalKernelTime, isSoftThresh);
	if (!bResult)
		return 1;

	OpenCLEnv::PrintProfilingInfo(totalKernelTime, "Matrix threshold");


	// ------------------------------------------------------------------------------------------------------
	// For all of the rows in the transposed matrix (i.e. column in the original one) invoke InverseHaarTransformGPU
	// ------------------------------------------------------------------------------------------------------
	totalKernelTime = 0;
	bResult = bResult && InverseHaarTransformGPU(gInBuff, gOutBuff, gPartialBuff, width, numLevelsHeight, height, 0, kernelTime);
	totalKernelTime += kernelTime;
	if (!bResult)
		return 1;

	OpenCLEnv::PrintProfilingInfo(totalKernelTime, "Inverse transform on columns");


	// ---------------------------------------------------------------------------------------
	// Transpose the matrix by invoking a kernel which will transpose the matrix on the device
	// w/o bringing it back to the host memory
	// ---------------------------------------------------------------------------------------
	bResult = TransposeMatrixGPU(gOutBuff, gInBuff, height, width, totalKernelTime);
	if (!bResult)
		return 1;

	OpenCLEnv::PrintProfilingInfo(totalKernelTime, "Matrix transpose");


	// -----------------------------------------------------------------
	// Invoke InverseHaarTransformGPU for all the rows simltaneously
	// -----------------------------------------------------------------
	totalKernelTime = 0;
	bResult = bResult && InverseHaarTransformGPU(gInBuff, gOutBuff, gPartialBuff, height, numLevelsWidth, width, 0, kernelTime);
	totalKernelTime += kernelTime;
	if (!bResult)
		return 1;
	
	OpenCLEnv::PrintProfilingInfo(totalKernelTime, "Inverse transform on rows");


	// -----------------------------------------------------------------
	// Read the results from the device
	// -----------------------------------------------------------------
	clErr = clEnqueueReadBuffer(m_oclEnv.m_cmdQ, gOutBuff, CL_TRUE, 0, gBuffSize, pInFloatsMatrix, 0, NULL, NULL);
	OpenCLEnv::CheckForError(clErr, "reading data from device");


	// ------------------------------------------------
	// Convert given buffer to a matrix of gray levels
	// ------------------------------------------------
	for (unsigned int i = 0; i < numPixels; i++)
		out[i] = (char)(pInFloatsMatrix[i] * 255.f);


	clReleaseMemObject(gInBuff);
	clReleaseMemObject(gOutBuff);
	clReleaseMemObject(gPartialBuff);


	return 0;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::PerformSelfTest()
{
	bool result1 = TestHaarTransformGPU();
	bool result2 = TestMatTransposeGPU();
	bool result3 = TestMatThreshGPU();

	return result1 && result2 && result3;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::TestMatTransposeGPU()
{
	const int TEMP_BUFF_SIZE = 512*512;
	float*		pTempBuff = new float[TEMP_BUFF_SIZE];
	float*		pResBuff = new float[TEMP_BUFF_SIZE];
	float*		pCorrectBuff = new float[TEMP_BUFF_SIZE];
	cl_int      clErr;
	cl_mem		gInBuff;
	cl_mem		gOutBuff;
	cl_ulong	matTransposeKernelTime = 0;

	int cnt = 0;
	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 512; j++)
			pTempBuff[i*512 + j] = (float)cnt++;

	cnt = 0;
	for (int i = 0; i < 512; i++)
		for (int j = 0; j < 512; j++)
			pCorrectBuff[j*512 + i] = (float)cnt++;

	unsigned int gBuffSize = TEMP_BUFF_SIZE * sizeof(float);
	gInBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_READ_ONLY, gBuffSize, NULL, NULL);
	gOutBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_WRITE_ONLY, gBuffSize, NULL, NULL);

	clErr = clEnqueueWriteBuffer(m_oclEnv.m_cmdQ, gInBuff, CL_TRUE, 0, gBuffSize, pTempBuff, 0, NULL, NULL);
	OpenCLEnv::CheckForError(clErr, "writing input buffer data to device");

	bool bResult = TransposeMatrixGPU(gInBuff, gOutBuff, 512, 512, matTransposeKernelTime);
	OpenCLEnv::PrintProfilingInfo(matTransposeKernelTime, "Matrix transpose");
	if (bResult)
	{
		clErr = clEnqueueReadBuffer(m_oclEnv.m_cmdQ, gOutBuff, CL_TRUE, 0, gBuffSize, pResBuff, 0, NULL, NULL);
		OpenCLEnv::CheckForError(clErr, "reading data from device");
		if (!OpenCLEnv::CompareFloatBuffers(pCorrectBuff, pResBuff, TEMP_BUFF_SIZE))
				bResult = false;
	}

	delete[] pTempBuff;
	delete[] pResBuff;
	delete[] pCorrectBuff;
	clReleaseMemObject(gInBuff);
	clReleaseMemObject(gOutBuff);

	return bResult;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::TestMatThreshGPU()
{
	const int TEMP_BUFF_SIZE = 5;
	float tempBuff[TEMP_BUFF_SIZE] = {0.f, -1.f, 2.f, -3.f, 4.f};
	float resBuff[TEMP_BUFF_SIZE];
	float correctBuff[TEMP_BUFF_SIZE] = {0.f, 0.f, 2.f, -3.f, 4.f};
	float thresh = 1.f;

	cl_int      clErr;
	cl_mem		gInBuff;
	cl_mem		gOutBuff;
	cl_ulong	matThreshKernelTime = 0;

	unsigned int gBuffSize = TEMP_BUFF_SIZE * sizeof(float);
	gInBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_READ_ONLY, gBuffSize, NULL, NULL);
	gOutBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_WRITE_ONLY, gBuffSize, NULL, NULL);

	clErr = clEnqueueWriteBuffer(m_oclEnv.m_cmdQ, gInBuff, CL_TRUE, 0, gBuffSize, tempBuff, 0, NULL, NULL);
	OpenCLEnv::CheckForError(clErr, "writing input buffer data to device");

	bool bResult = MatrixThreshGPU(gInBuff, gOutBuff, TEMP_BUFF_SIZE, thresh, matThreshKernelTime);
	OpenCLEnv::PrintProfilingInfo(matThreshKernelTime, "Matrix thresh");
	if (bResult)
	{
		clErr = clEnqueueReadBuffer(m_oclEnv.m_cmdQ, gOutBuff, CL_TRUE, 0, gBuffSize, resBuff, 0, NULL, NULL);
		OpenCLEnv::CheckForError(clErr, "reading data from device");
		if (!OpenCLEnv::CompareFloatBuffers(correctBuff, resBuff, TEMP_BUFF_SIZE))
				bResult = false;
	}

	clReleaseMemObject(gInBuff);
	clReleaseMemObject(gOutBuff);

	return bResult;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::TestHaarTransformGPU()
{
	float* pInBuff = NULL;
	float* pOutBuff = NULL;
	float* pRefData = NULL;
	float* pInvRefData = NULL;
	unsigned int globalOffset = 0;
	unsigned int buffLen = 0;
	unsigned int lenRef = 0;
	unsigned int invLenRef = 0;
	bool result = true;
	if (OpenCLEnv::ReadFileFloat(TEST_SIGNAL_FILE_1, &pInBuff, &buffLen))
	{
		unsigned int numLevels = 0;
		if (!CNoiseCleaner::GetNumLevels(buffLen, numLevels))
			return false;	// The buffer length is not a power of two

		pOutBuff = new float[buffLen];
		pInvRefData = new float[buffLen];

		cl_int                  clErr;
		cl_mem					gInBuff;
		cl_mem					gOutBuff;
		cl_mem					gPartialBuff;
		cl_ulong				totalFWTKernelTime = 0;
		cl_ulong				totalIWTKernelTime = 0;

		// -----------------------------------------
		// Allocate GPU buffers and send data to GPU
		// -----------------------------------------
		unsigned int gBuffSize = buffLen * sizeof(float);
		gInBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_READ_WRITE, gBuffSize, NULL, NULL);
		gOutBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_READ_WRITE, gBuffSize, NULL, NULL);
		gPartialBuff = clCreateBuffer(m_oclEnv.m_context, CL_MEM_WRITE_ONLY, gBuffSize, NULL, NULL);

		clErr = clEnqueueWriteBuffer(m_oclEnv.m_cmdQ, gInBuff, CL_TRUE, 0, gBuffSize, pInBuff, 0, NULL, NULL);
		OpenCLEnv::CheckForError(clErr, "writing input buffer data to device");
	
		if (!ForwardHaarTransformGPU(gInBuff, gOutBuff, gPartialBuff, 1, numLevels, buffLen, globalOffset, totalFWTKernelTime))
			result = false;
		OpenCLEnv::PrintProfilingInfo(totalFWTKernelTime, "ForwardHaarTransformGPU");
		if (result)
		{
			clErr = clEnqueueReadBuffer(m_oclEnv.m_cmdQ, gOutBuff, CL_TRUE, 0, gBuffSize, pOutBuff, 0, NULL, NULL);
			OpenCLEnv::CheckForError(clErr, "reading data from device");

			if (OpenCLEnv::ReadFileFloat(TEST_REGRESS_FILE_1, &pRefData, &lenRef))
			{
				if (lenRef != buffLen || !OpenCLEnv::CompareFloatBuffers(pOutBuff, pRefData, buffLen))
					result = false;
				if (result)
				{
					if (!InverseHaarTransformGPU(gOutBuff, gInBuff, gPartialBuff, 1, numLevels, buffLen, globalOffset, totalIWTKernelTime))
						result = false;
					OpenCLEnv::PrintProfilingInfo(totalIWTKernelTime, "InverseHaarTransformGPU");
					if (result)
					{
						clErr = clEnqueueReadBuffer(m_oclEnv.m_cmdQ, gInBuff, CL_TRUE, 0, gBuffSize, pInvRefData, 0, NULL, NULL);
						OpenCLEnv::CheckForError(clErr, "reading data from device");
						if (!OpenCLEnv::CompareFloatBuffers(pInBuff, pInvRefData, buffLen))
							result = false;
					}
				}
			}
		}

		delete[] pInBuff;
		delete[] pOutBuff;
		delete[] pRefData;
		delete[] pInvRefData;
		clReleaseMemObject(gInBuff);
		clReleaseMemObject(gOutBuff);
		clReleaseMemObject(gPartialBuff);
	}
	return result;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::TestHaarTransformCPU()
{
	float* pInData = NULL;
	float* pOutData = NULL;
	float* pRefData = NULL;
	float* pInvRefData = NULL;
	unsigned int len = 0;
	unsigned int lenRef = 0;
	unsigned int invLenRef = 0;
	bool result = true;
	if (OpenCLEnv::ReadFileFloat(TEST_SIGNAL_FILE_1, &pInData, &len))
	{
		pOutData = new float[len];
		pInvRefData = new float[len];
		CNoiseCleaner::ForwardHaarTransformCPU(pInData, len, pOutData, 0);
		
		if (OpenCLEnv::ReadFileFloat(TEST_REGRESS_FILE_1, &pRefData, &lenRef))
		{
			if (lenRef != len || !OpenCLEnv::CompareFloatBuffers(pOutData, pRefData, len))
				result = false;
			if (result)
			{
				CNoiseCleaner::InverseHaarTransformCPU(pRefData, len, pInvRefData, 0);
				if (!OpenCLEnv::CompareFloatBuffers(pInData, pInvRefData, len))
					result = false;
			}
		}
		
		delete[] pInData;
		delete[] pOutData;
		delete[] pRefData;
		delete[] pInvRefData;
	}

	return result;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::ForwardHaarTransformGPU(cl_mem gInBuff, cl_mem gOutBuff, cl_mem gPartialBuff, int numGroups,
											unsigned int numLevels, unsigned int dataLen, unsigned int globalOffset, cl_ulong& kernelTime)
{
	cl_int                  clErr;
	cl_event                kernelEvent;
	cl_ulong				totalKernelTime = 0;

	unsigned int numThreadsLeft = dataLen >> 1;
	unsigned int numLevelsLeft = numLevels;
	unsigned int globalOffsetByChars = globalOffset * sizeof(float);

	// Number of levels on device are determined by the work-group size
	size_t localWorkItems;
	size_t globalWorkItems;
	unsigned int maxLevelsOnDevice = 0;
	unsigned int currLevels = 0;
	CNoiseCleaner::GetNumLevels(m_oclEnv.m_kernelWorkGroupSizes[FWT_KERNEL_IDX], maxLevelsOnDevice);
	maxLevelsOnDevice++;

	while (numThreadsLeft > 0)
	{
		currLevels = numLevelsLeft < maxLevelsOnDevice ? numLevelsLeft : maxLevelsOnDevice;
		localWorkItems = 1 << (currLevels - 1);
		globalWorkItems = numThreadsLeft*numGroups;
		// Each thread stores two floats in local memory
		unsigned int locMemSize = localWorkItems * 2 * sizeof(cl_float);

		// Set arguments 
		clSetKernelArg(m_oclEnv.m_kernels[FWT_KERNEL_IDX], 0, sizeof(cl_mem), &gInBuff);
		clSetKernelArg(m_oclEnv.m_kernels[FWT_KERNEL_IDX], 1, sizeof(cl_mem), &gOutBuff);
		clSetKernelArg(m_oclEnv.m_kernels[FWT_KERNEL_IDX], 2, sizeof(cl_mem), &gPartialBuff);
		clSetKernelArg(m_oclEnv.m_kernels[FWT_KERNEL_IDX], 3, locMemSize, NULL);
		clSetKernelArg(m_oclEnv.m_kernels[FWT_KERNEL_IDX], 4, sizeof(unsigned int), &currLevels);
		clSetKernelArg(m_oclEnv.m_kernels[FWT_KERNEL_IDX], 5, sizeof(unsigned int), &globalOffset);
		clSetKernelArg(m_oclEnv.m_kernels[FWT_KERNEL_IDX], 6, sizeof(unsigned int), &dataLen);
		
		// Run kernel
		clErr = clEnqueueNDRangeKernel(m_oclEnv.m_cmdQ, m_oclEnv.m_kernels[FWT_KERNEL_IDX], 1, NULL, &globalWorkItems, &localWorkItems, 0, NULL, &kernelEvent);
		OpenCLEnv::CheckForError(clErr, "enqueuing FWT kernel");
		
		clErr = clWaitForEvents(1, &kernelEvent);
		OpenCLEnv::CheckForError(clErr, "wait for kernel to finish");

		totalKernelTime += OpenCLEnv::GetKernelTime(kernelEvent);

		numLevelsLeft -= currLevels;
		numThreadsLeft >>= currLevels;

		clReleaseEvent(kernelEvent);
	}
	kernelTime = totalKernelTime;

	return true;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::InverseHaarTransformGPU(cl_mem gInBuff, cl_mem gOutBuff, cl_mem gPartialBuff, int numGroups,
											unsigned int numLevels, unsigned int dataLen, unsigned int globalOffset, cl_ulong& kernelTime)
{
	cl_int                  clErr;
	cl_event                kernelEvent;
	cl_event                bufferSyncEvent;
	cl_ulong				totalKernelTime = 0;

	unsigned int numThreadsLeft = dataLen >> 1;
	unsigned int numLevelsLeft = numLevels;
	unsigned int globalOffsetByChars = globalOffset * sizeof(float);

	// Number of levels on device are determined by the work-group size
	size_t localWorkItems;
	size_t globalWorkItems;
	unsigned int maxLevelsOnDevice = 0;
	unsigned int currLevels = 0;
	CNoiseCleaner::GetNumLevels(m_oclEnv.m_kernelWorkGroupSizes[IWT_KERNEL], maxLevelsOnDevice);
	maxLevelsOnDevice++;

	// Activate first stage kernel
	currLevels = numLevelsLeft < maxLevelsOnDevice ? numLevelsLeft : maxLevelsOnDevice;
	localWorkItems = 1 << (currLevels - 1);
	globalWorkItems = localWorkItems*numGroups;
	// Each thread stores two floats in local memory
	unsigned int locMemSize = localWorkItems * 2 * sizeof(cl_float);
	// Set arguments 
	clSetKernelArg(m_oclEnv.m_kernels[IWT_KERNEL], 0, sizeof(cl_mem), &gInBuff);
	clSetKernelArg(m_oclEnv.m_kernels[IWT_KERNEL], 1, locMemSize, NULL);
	clSetKernelArg(m_oclEnv.m_kernels[IWT_KERNEL], 2, locMemSize, NULL);
	clSetKernelArg(m_oclEnv.m_kernels[IWT_KERNEL], 3, sizeof(cl_float), NULL);
	clSetKernelArg(m_oclEnv.m_kernels[IWT_KERNEL], 4, sizeof(unsigned int), &currLevels);
	clSetKernelArg(m_oclEnv.m_kernels[IWT_KERNEL], 5, sizeof(unsigned int), &globalOffset);
	clSetKernelArg(m_oclEnv.m_kernels[IWT_KERNEL], 6, sizeof(unsigned int), &dataLen);
		
	// Run kernel
	clErr = clEnqueueNDRangeKernel(m_oclEnv.m_cmdQ, m_oclEnv.m_kernels[IWT_KERNEL], 1, NULL, &globalWorkItems, &localWorkItems, 0, NULL, &kernelEvent);
	OpenCLEnv::CheckForError(clErr, "enqueuing FWT kernel");
		
	clErr = clWaitForEvents(1, &kernelEvent);
	OpenCLEnv::CheckForError(clErr, "wait for kernel to finish");

	totalKernelTime += OpenCLEnv::GetKernelTime(kernelEvent);

	numLevelsLeft -= currLevels;
	numThreadsLeft = 1 << currLevels;
		
	bool switchBuffers = false;

	if (!switchBuffers)
	{
		clErr = clEnqueueCopyBuffer(m_oclEnv.m_cmdQ, gInBuff, gOutBuff, globalOffsetByChars, globalOffsetByChars, numGroups*dataLen*sizeof(float), 0, NULL, &bufferSyncEvent);
		OpenCLEnv::CheckForError(clErr, "copy buffers inside device");
		clErr = clWaitForEvents(1, &bufferSyncEvent);
		OpenCLEnv::CheckForError(clErr, "wait for buffer copy inside device");

		clReleaseEvent(bufferSyncEvent);
	}
	kernelTime = totalKernelTime;

	return true;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::TransposeMatrixGPU(cl_mem gInBuff, cl_mem gOutBuff, int width, int height, cl_ulong& kernelTime)
{
	cl_int                  clErr;
	cl_event                kernelEvent;

	unsigned int locMemSize = TILE_SIZE * TILE_SIZE * sizeof(cl_float);
	clSetKernelArg(m_oclEnv.m_kernels[MAT_TRANSPOSE_KERNEL], 0, sizeof(cl_mem), &gInBuff);
	clSetKernelArg(m_oclEnv.m_kernels[MAT_TRANSPOSE_KERNEL], 1, sizeof(cl_mem), &gOutBuff);
	clSetKernelArg(m_oclEnv.m_kernels[MAT_TRANSPOSE_KERNEL], 2, locMemSize, NULL);
	clSetKernelArg(m_oclEnv.m_kernels[MAT_TRANSPOSE_KERNEL], 3, sizeof(unsigned int), &width);
	clSetKernelArg(m_oclEnv.m_kernels[MAT_TRANSPOSE_KERNEL], 4, sizeof(unsigned int), &height);

	size_t localWorkItems[2] = {TILE_SIZE, TILE_SIZE};
	size_t globalWorkItems[2];
	globalWorkItems[0] = ((width - 1) / localWorkItems[0] + 1) * localWorkItems[0];
	globalWorkItems[1] = ((height - 1) / localWorkItems[1] + 1) * localWorkItems[1];
	clErr = clEnqueueNDRangeKernel(m_oclEnv.m_cmdQ, m_oclEnv.m_kernels[MAT_TRANSPOSE_KERNEL], 2, NULL, globalWorkItems, localWorkItems, 0, NULL, &kernelEvent);
	OpenCLEnv::CheckForError(clErr, "enqueuing transpose kernel");

	clErr = clWaitForEvents(1, &kernelEvent);
	OpenCLEnv::CheckForError(clErr, "wait for kernel to finish");

	kernelTime = OpenCLEnv::GetKernelTime(kernelEvent);
	clReleaseEvent(kernelEvent);

	return true;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::MatrixThreshGPU(cl_mem gInBuff, cl_mem gOutBuff, unsigned int dataLen, float thresh, cl_ulong& kernelTime, bool isSoftThresh /*= false*/)
{
	cl_int                  clErr;
	cl_event                kernelEvent;

	int kernelIdx = MAT_HT_THRESH_KERNEL;
	if (isSoftThresh)
		kernelIdx = MAT_ST_THRESH_KERNEL;

	clSetKernelArg(m_oclEnv.m_kernels[kernelIdx], 0, sizeof(cl_mem), &gInBuff);
	clSetKernelArg(m_oclEnv.m_kernels[kernelIdx], 1, sizeof(cl_mem), &gOutBuff);
	clSetKernelArg(m_oclEnv.m_kernels[kernelIdx], 2, sizeof(float), &thresh);

	size_t localWorkItems = 256;
	size_t globalWorkItems = ((dataLen - 1) / localWorkItems + 1) * localWorkItems;
	clErr = clEnqueueNDRangeKernel(m_oclEnv.m_cmdQ, m_oclEnv.m_kernels[kernelIdx], 1, NULL, &globalWorkItems, &localWorkItems, 0, NULL, &kernelEvent);
	OpenCLEnv::CheckForError(clErr, "enqueuing matrix thresh kernel");

	clErr = clWaitForEvents(1, &kernelEvent);
	OpenCLEnv::CheckForError(clErr, "wait for kernel to finish");

	kernelTime = OpenCLEnv::GetKernelTime(kernelEvent);
	clReleaseEvent(kernelEvent);

	return true;
}
//-----------------------------------------------------------------------------------------
bool CNoiseCleaner::GetNumLevels(unsigned int buffLen, unsigned int& numLevels)
{
	numLevels = (unsigned int)(log((double)buffLen) / log(2.0));
	return ((1 << numLevels) == buffLen);
}
//-----------------------------------------------------------------------------------------
void CNoiseCleaner::ForwardHaarTransformCPU(const float* pInBuff, unsigned int buffLen, float* pOutBuff, unsigned int globalOffset)
{
	int i = 0;
	int w = buffLen;
	float* pTempIn = new float[buffLen];
	memcpy(pTempIn, pInBuff, buffLen * sizeof(float));

#if defined(_WIN32) || defined(_WIN64) || defined(_WINDOWS)
	LARGE_INTEGER perfFreq;
	LARGE_INTEGER perfCounterStart;
	QueryPerformanceFrequency(&perfFreq);
	QueryPerformanceCounter(&perfCounterStart);
#endif

	while (w > 1)
	{
		w /= 2;
		for (i = 0; i < w; i++)
		{
			pOutBuff[i] = (pTempIn[2*i] + pTempIn[2*i + 1]) * INV_SQRT_2;
			pOutBuff[i+w] = (pTempIn[2*i] - pTempIn[2*i + 1]) * INV_SQRT_2;
		}
		memcpy(pTempIn, pOutBuff, w * 2 * sizeof(float));
	}

#if defined(_WIN32) || defined(_WIN64) || defined(_WINDOWS)
	LARGE_INTEGER perfCounterEnd;
	QueryPerformanceCounter(&perfCounterEnd);
	double funcTimeInMicros = ((double)(perfCounterEnd.QuadPart - perfCounterStart.QuadPart) / (double)perfFreq.QuadPart) * 10e6;
	std::cout << "ForwardHaarTransformCPU ran for: " << funcTimeInMicros << " micro sec\n";
#endif

	delete[] pTempIn;
}
//-----------------------------------------------------------------------------------------
void CNoiseCleaner::InverseHaarTransformCPU(const float* pInBuff, unsigned int buffLen, float* pOutBuff, unsigned int globalOffset)
{
	unsigned int i = 0;
	unsigned int w = 1;
	float* pTempIn = new float[buffLen];
	memcpy(pTempIn, pInBuff, buffLen * sizeof(float));

#if defined(_WIN32) || defined(_WIN64) || defined(_WINDOWS)
	LARGE_INTEGER perfFreq;
	LARGE_INTEGER perfCounterStart;
	QueryPerformanceFrequency(&perfFreq);
	QueryPerformanceCounter(&perfCounterStart);
#endif

	while (w < buffLen)
	{
		for (i = 0; i < w; i++)
		{
			pOutBuff[2*i] = (pTempIn[i] + pTempIn[i+w]) * SQRT_2 * 0.5f;
			pOutBuff[2*i + 1] = (pTempIn[i] * SQRT_2) - pOutBuff[2*i];
		}
		memcpy(pTempIn, pOutBuff, w * 2 * sizeof(float));
		w *= 2;
	}

#if defined(_WIN32) || defined(_WIN64) || defined(_WINDOWS)
	LARGE_INTEGER perfCounterEnd;
	QueryPerformanceCounter(&perfCounterEnd);
	double funcTimeInMicros = ((double)(perfCounterEnd.QuadPart - perfCounterStart.QuadPart) / (double)perfFreq.QuadPart) * 10e6;
	std::cout << "ForwardHaarTransformCPU ran for: " << funcTimeInMicros << " micro sec\n";
#endif

	delete[] pTempIn;
}
//-----------------------------------------------------------------------------------------

