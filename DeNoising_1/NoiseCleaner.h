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

#ifndef __NOISE_CLEANER_H__
#define __NOISE_CLEANER_H__


#include <CL/cl.h>


// -----------------------------------------------------------------------------------------
// This class encapsulates the logic of GPU-based DeNoising. It uses OpenCL
// to accelerate the algorithm and thus capable of running on both NVIDIA 
// and AMD devices.
// The class is used by first constructing an instance and then activating the
// 'CleanNoise' method. The construction phase reads the required kernels from
// the accompanying 'HWT_kernels.cl' file and compiles them. This results in a
// slight delay, but once an instance is constructed it can be used throughout
// the application many times without recompiling the kernels.
// -----------------------------------------------------------------------------------------
class CNoiseCleaner
{
public:
	CNoiseCleaner();
	~CNoiseCleaner();

	// -----------------------------------------------------------------------------------------
	// This method performs the actual 'DeNoising' algorithm on the given 'in' matrix which is
	// assumed to be a 1-channel (grayscale) signal. The result is stored in 'out' matrix which
	// have to be allocated and has to have exactly the same size as 'in'. This method is designed
	// to resemble WaveLab's 'ThreshWave2' function and it consists of 3 stages:
	// 1. Run forward Haar transform on the input
	// 2. Use threshold to filter wavelet coefficients (the values resulting from previous stage)
	// 3. Run inverse Haar transform on the filtered wavelet coefficients.
	// 'width', 'height' - Specify the size of the 'in' matrix, it is assumed that 'width'=2^J and
	//					   'height'= 2^K, in other words, width and height are equal to some power of 2
	//						and it doesn't have to be same power. However the maximal possible size of
	//						width and height is limited to 1024.
	// 'thresh' - Specifies the threshold to use during the 2nd stage.
	// 'isSoftThresh' - If this value is true then 'soft threshold' is used, otherwise 'hard threshold' is
	//					used in the 2nd stage. The behaviour of this two thresholding techniques is exactly 
	//					the same as in WaveLab's 'ThreshWave2' function (which is part of DeNoising package).
	// -----------------------------------------------------------------------------------------
	int CleanNoise(unsigned char *in, unsigned char *out, int width, int height, float thresh, bool isSoftThresh);


	// -----------------------------------------------------------------------------------------
	// Performs an internal test of OpenCL kernels using signals from accompanying external files.
	// This method is useful to validate this class.
	// -----------------------------------------------------------------------------------------
	bool PerformSelfTest();

	
private:
	enum KernelIndices
	{
		FWT_KERNEL_IDX, IWT_KERNEL, MAT_TRANSPOSE_KERNEL, MAT_HT_THRESH_KERNEL, MAT_ST_THRESH_KERNEL, NUM_KERNELS
	};
	OpenCLEnv	m_oclEnv;

	/** Each one of this method activates OpenCL kernels with the given parameters and leaves the results on the GPU **/
	bool ForwardHaarTransformGPU(cl_mem gInBuff, cl_mem gOutBuff, cl_mem gPartialBuff, int numGroups,
								 unsigned int numLevels, unsigned int dataLen, unsigned int globalOffset, cl_ulong& kernelTime);
	bool InverseHaarTransformGPU(cl_mem gInBuff, cl_mem gOutBuff, cl_mem gPartialBuff, int numGroups,
							     unsigned int numLevels, unsigned int dataLen, unsigned int globalOffset, cl_ulong& kernelTime);
	bool TransposeMatrixGPU(cl_mem gInBuff, cl_mem gOutBuff, int width, int height, cl_ulong& kernelTime);
	bool MatrixThreshGPU(cl_mem gInBuff, cl_mem gOutBuff, unsigned int dataLen, float thresh, cl_ulong& kernelTime, bool isSoftThresh = false);

	static bool GetNumLevels(unsigned int buffLen, unsigned int& numLevels);
	static char* KERNEL_NAMES[NUM_KERNELS];

	/** Auxiliary methods for testing GPU kernels **/
	bool TestHaarTransformGPU();
	bool TestMatTransposeGPU();
	bool TestMatThreshGPU();

	/** CPU routines for testing **/
	static bool TestHaarTransformCPU();
	static void ForwardHaarTransformCPU(const float* pInBuff, unsigned int buffLen, float* pOutBuff, unsigned int globalOffset);
	static void InverseHaarTransformCPU(const float* pInBuff, unsigned int buffLen, float* pOutBuff, unsigned int globalOffset);
};



#endif	// __NOISE_CLEANER_H__
