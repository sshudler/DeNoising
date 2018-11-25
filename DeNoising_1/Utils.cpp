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

#include "Utils.h"
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

//-----------------------------------------------------------------------------------------
void OpenCLEnv::ReadFileToString(const char *filename, char **fileString)
{   
    size_t size;
    char *buf = NULL;
    
	std::ifstream file (filename, std::ios::in | std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        *fileString = NULL;
        return;
    }
    
    size = file.tellg();
    buf = new char[size + 1];
    file.seekg(0, std::ios::beg);
    file.read(buf, size);
    buf[size] = 0;
    file.close();
    
    *fileString = buf;
}
//-----------------------------------------------------------------------------------------
void OpenCLEnv::PrintProfilingInfo(cl_event kernelEvent, const char* pKernelName)
{
	cl_ulong kernelTime = GetKernelTime(kernelEvent);
	PrintProfilingInfo(kernelTime, pKernelName);
}
//-----------------------------------------------------------------------------------------
void OpenCLEnv::PrintProfilingInfo(cl_ulong totalKernelTime, const char* pKernelName)
{
	std::cout << pKernelName << " ran for: " << (double)(totalKernelTime)/(double)10e3 << " micro sec\n";
}
//-----------------------------------------------------------------------------------------
cl_ulong OpenCLEnv::GetKernelTime(cl_event kernelEvent)
{
	cl_int 		clErr;
	cl_ulong	kernelStartTime;
	cl_ulong	kernelEndTime;

	clErr = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelStartTime, NULL);
	OpenCLEnv::CheckForError(clErr, "getting kernel profiling info");
	clErr = clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelEndTime, NULL);
	OpenCLEnv::CheckForError(clErr, "getting kernel profiling info");
	return (kernelEndTime - kernelStartTime);
}
//-----------------------------------------------------------------------------------------
bool OpenCLEnv::ReadFileFloat(const char* filename, float** data, unsigned int* len)
{
	if (filename == NULL || len == NULL || data == NULL)
		return false;

	std::vector<float>	dataRead;
	std::fstream		fh(filename, std::fstream::in);

	if (!fh.good())
		return false;

	float token;
	while (fh.good())
	{
		fh >> token;
		dataRead.push_back(token);
	}

	dataRead.pop_back();
	fh.close();

	if (*data != NULL)
		return false;

	*data = new float[dataRead.size()];
	*len = dataRead.size();

	memcpy(*data, &dataRead.front(), sizeof(float) * (*len));

	return true;
}
//-----------------------------------------------------------------------------------------
bool OpenCLEnv::WriteFileFloat(const char* filename, const float* data, unsigned int len)
{
	if (filename == NULL || data == NULL)
		return false;

	std::fstream		fh(filename, std::fstream::out);

	if (!fh.good())
		return false;

	fh << std::fixed << std::setprecision(10);
	for (unsigned int i = 0; i < len; i++)
		fh << data[i] << std::endl;

	fh.close();

	return true;
}
//-----------------------------------------------------------------------------------------
bool OpenCLEnv::CompareFloatBuffers(const float* pInBuff1, const float* pInBuff2, unsigned int buffLen)
{
	const float EPSILON = 0.001f;
	for (unsigned int i = 0; i < buffLen; i++)
	{
		if (fabs(pInBuff1[i] - pInBuff2[i]) > EPSILON)
			return false;
	}
	return true;
}
//-----------------------------------------------------------------------------------------
OpenCLEnv::OpenCLEnv(const char* pFilename, int numKernels, char** pKernelNames)
{
	cl_uint				numPlatforms = 0;
	cl_platform_id*		platformIDs = NULL;
	cl_int				clErr;
	cl_bool				supportsImages;
	char*				pOCLKernelsStr = NULL;


	//
	// Have to find a GPU device by querying each available platform 
	//
	clErr = clGetPlatformIDs(0, NULL, &numPlatforms);
	CheckForError(clErr, "querying platforms");
	platformIDs = new cl_platform_id[numPlatforms];
	clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	unsigned int i = 0;
	clErr = clGetDeviceIDs(platformIDs[i++], CL_DEVICE_TYPE_GPU, 1, &m_deviceID, NULL);
	while (clErr != CL_SUCCESS && i < numPlatforms)
	    clErr = clGetDeviceIDs(platformIDs[i++], CL_DEVICE_TYPE_GPU, 1, &m_deviceID, NULL);
	CheckForError(clErr, "querying for device");
	delete[] platformIDs;	// platformIDs no longer needed

	// 
	// Check whether the device supports images 
	//
	clErr = clGetDeviceInfo(m_deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &supportsImages, NULL);
	CheckForError(clErr, "querying for image support");
	m_isSupportsImages = (bool)supportsImages;
	
	//
	// Create context and command queue; both are needed for running kernels
	//
	m_context = clCreateContext(0, 1, &m_deviceID, NULL, NULL, &clErr);  
	CheckForError(clErr, "creating context");
	m_cmdQ = clCreateCommandQueue(m_context, m_deviceID, CL_QUEUE_PROFILING_ENABLE, &clErr); 
	CheckForError(clErr, "creating command queue");

	//
	// Read the kernels from disk
	//
	ReadFileToString(pFilename, &pOCLKernelsStr);
	CheckForError(pOCLKernelsStr ? CL_SUCCESS : CL_INVALID_BINARY, "opening kernel file");
	m_program = clCreateProgramWithSource(m_context, 1, (const char **) &pOCLKernelsStr, NULL, &clErr); 
	CheckForError(clErr, "creating program");

	//
	// Compile the program (print the build log if needed) and query for the kernel
	//
	clErr = clBuildProgram(m_program, 0, NULL, NULL, NULL, NULL);
	if (clErr != CL_SUCCESS) {
		std::cout << "OpenCL error: building program (" << clErr << ")!" << std::endl;
		size_t buildLogSize = 0;
		char* pBuildLog = NULL;
		clGetProgramBuildInfo(m_program, m_deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
		pBuildLog = new char[buildLogSize + 1];
		clGetProgramBuildInfo(m_program, m_deviceID, CL_PROGRAM_BUILD_LOG, buildLogSize, pBuildLog, NULL);
		pBuildLog[buildLogSize] = '\0';
		std::cout << "Build log:" << std::endl << pBuildLog << std::endl;
		std::cout << "Press enter to exit\n";
		getchar();
		delete[] pBuildLog;
		exit(1);
	}
	delete[] pOCLKernelsStr;	// the kernel source no longer needed 
	
	m_numKernels = numKernels;
	m_kernels = new cl_kernel[m_numKernels];
	for (int i = 0; i < m_numKernels; i++)
	{
		m_kernels[i] = clCreateKernel(m_program, pKernelNames[i], &clErr);
		CheckForError(clErr, "querying for kernel");
	}

	//
	// Get max workgroup size
	//
	m_kernelWorkGroupSizes = new unsigned int[m_numKernels];
	for (int i = 0; i < m_numKernels; i++)
	{
		clErr = clGetKernelWorkGroupInfo(m_kernels[i], m_deviceID, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
										 &m_kernelWorkGroupSizes[i], 0);
		CheckForError(clErr, "querying for kernel");
	}
	
}
//-----------------------------------------------------------------------------------------
OpenCLEnv::~OpenCLEnv()
{
	for (int i = 0; i < m_numKernels; i++)
		clReleaseKernel(m_kernels[i]);
	
	delete[] m_kernels;
	delete[] m_kernelWorkGroupSizes;
	clReleaseProgram(m_program); 
	clReleaseCommandQueue(m_cmdQ); 
	clReleaseContext(m_context);
}
//-----------------------------------------------------------------------------------------
