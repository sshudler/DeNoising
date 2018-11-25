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

#ifndef __UTILS_H__
#define __UTILS_H__

#include <CL/cl.h>
#include <stdlib.h>
#include <iostream>



// ----------------------------------------------------------------------------
// Helper class for initializing the OpenCL environment
// ----------------------------------------------------------------------------
class OpenCLEnv
{
public:
    inline static void CheckForError(cl_int clErr, const char* message)
    {
    	if (clErr != CL_SUCCESS)
    	{
    		std::cerr << "OpenCL error: " << message << " " << clErr << std::endl;
    		exit(1);
    	}
    }

	// ----------------------------------------------------------------------------
	// Helper function to read kernel files (.cl) from disk
	// ----------------------------------------------------------------------------
    static void ReadFileToString(const char* filename, char** fileString);


	// ----------------------------------------------------------------------------
	// Helper functions to print profiling info
	// ----------------------------------------------------------------------------
    static void PrintProfilingInfo(cl_event kernelEvent, const char* pKernelName);
	static void PrintProfilingInfo(cl_ulong totalKernelTime, const char* pKernelName);


	// ----------------------------------------------------------------------------
	// Helper function to extract kernel time
	// ----------------------------------------------------------------------------
	static cl_ulong GetKernelTime(cl_event event);

	// ----------------------------------------------------------------------------
	// Helper function to read generic float file
	// ----------------------------------------------------------------------------
	static bool ReadFileFloat(const char* filename, float** data, unsigned int* len);

	// ----------------------------------------------------------------------------
	// Helper function to write generic float file
	// ----------------------------------------------------------------------------
	static bool WriteFileFloat(const char* filename, const float* data, unsigned int len);

	// ----------------------------------------------------------------------------
	// Helper function to compare two float buffers
	// ----------------------------------------------------------------------------
	static bool CompareFloatBuffers(const float* pInBuff1, const float* pInBuff2, unsigned int buffLen);

	cl_device_id		m_deviceID;
	cl_context			m_context; 
	cl_command_queue	m_cmdQ;
	cl_program			m_program;
	int					m_numKernels;
	cl_kernel*			m_kernels;
	unsigned int*		m_kernelWorkGroupSizes;
	bool				m_isSupportsImages;

	OpenCLEnv(const char* pFilename, int numKernels, char** pKernelNames);
	~OpenCLEnv();
};


#endif		// __UTILS_H__
