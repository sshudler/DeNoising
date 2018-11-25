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

#define INV_SQRT_2      0.70710678118654752440f
#define SQRT_2			1.41421356237309504880f
#define LOG_NUM_BANKS	4
#define BLOCK_ROWS		8


//
// This is the kernel for the 1D forward Haar wavelet transform
//
__kernel void FWT_kernel(__global float* inBuff, __global float* outBuff, __global float* partialBuff,
						 __local float* localBuff, const uint levels, const uint buffOffset, const uint buffLen)
{
	uint localId = get_local_id(0);
    uint groupId = get_group_id(0);
    uint localSize = get_local_size(0);
   
    uint buffOffset1 = buffOffset + groupId*buffLen;
    
    localBuff[localId] = inBuff[buffOffset1 + localId];
    localBuff[localId + localSize] = inBuff[buffOffset1 + localId + localSize];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // This will result in a bank conflict
	float data0 = localBuff[2 * localId];
	float data1 = localBuff[2 * localId + 1];
	
	// Detail coefficient, not further referenced in this kernel so directly store in global memory
	outBuff[buffOffset1 + localId + localSize] = (data0 - data1) * INV_SQRT_2;
	
	// Approximation coefficient: store in local memory for further decomposition steps in this global step
    localBuff[localId] = (data0 + data1) * INV_SQRT_2;
    
    // All threads have to write approximation coefficient to shared memory before 
    // next steps can take place
    barrier(CLK_LOCAL_MEM_FENCE);
    
	// Offset to second element in local element which has to be used for the 
	// decomposition, effectively 2^(i - 1)
	uint offsetNeighbor = 1;
	uint activeThreads = localSize >> 1;
	uint midOutPos = localSize >> 1;
	// Index for the first element of the pair to process, the representation is still compact (and therefore
	// still localId * 2) because the first step operated on registers and only the result has been written
	// to shared memory
	uint idata0 = localId * 2;
	
	for(uint i = 1; i < levels; ++i)
    {
		if  (localId >= activeThreads)
			return;
			
		// Update stride, with each decomposition level the stride grows by a factor of 2
		uint idata1 = idata0 + offsetNeighbor;
						
		// position of write into global memory
		uint globalWritePos = midOutPos + localId;
			
		data0 = localBuff[idata0];
		data1 = localBuff[idata1];
			
		// Detail coefficient, not further referenced in this kernel so directly store in global memory
		outBuff[buffOffset1 + globalWritePos] = (data0 - data1) * INV_SQRT_2;
			
		// Approximation coefficient: store in local memory for further decomposition steps in this global step
		localBuff[idata0] = (data0 + data1) * INV_SQRT_2;
				
		midOutPos >>= 1;
		activeThreads >>= 1;
		offsetNeighbor <<= 1;
		idata0 <<= 1;
                
        barrier(CLK_LOCAL_MEM_FENCE); 
    }
    
	// Write the top most level element for the next decomposition steps
	// which are performed after an interblock syncronization on host side
	if (0 == localId) 
	{
		//partialBuff[groupId] = localBuff[0];
		outBuff[buffOffset1] = localBuff[0];
	}
}


//
// This is the kernel for the 1D inverse Haar wavelet transform
//
__kernel void IWT_kernel(__global float* inBuff, __local float* localBuff, __local float* localBuff1, 
								   __local float* flipBuffFlag, const uint levels, const uint buffOffset, const uint buffLen)
{
	uint localId = get_local_id(0);
    uint localSize = get_local_size(0);
    uint groupId = get_group_id(0);
    
    uint buffOffset1 = buffOffset + groupId*buffLen;
    
    float data0 = inBuff[buffOffset1 + localId];
    float data1 = inBuff[buffOffset1 + localId + localSize];
    localBuff[localId] = data0;
    localBuff[localId + localSize] = data1;
    localBuff1[localId] = data0;
    localBuff1[localId + localSize] = data1;
    *flipBuffFlag = 0.f;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    uint activeThreads = 1;
	float res = 0.f;
         
    for(uint i = 0; i < levels; ++i)
    {
		if(localId < activeThreads)
        {
			uint i_idata0 = localId; 
			uint i_idata1 = localId+activeThreads; 
			uint o_idata0 = localId << 1; 
			uint o_idata1 = o_idata0 + 1; 
			
			if (*flipBuffFlag > 0.f)
			{
				data0 = localBuff1[i_idata0];
				data1 = localBuff1[i_idata1];
				res = (data0 + data1) * SQRT_2 * 0.5f;
				localBuff[o_idata0] = res;
				localBuff[o_idata1] = (data0 * SQRT_2) - res;
			}
			else
			{
				data0 = localBuff[i_idata0];
				data1 = localBuff[i_idata1];
				res = (data0 + data1) * SQRT_2 * 0.5f;
				localBuff1[o_idata0] = res;
				localBuff1[o_idata1] = (data0 * SQRT_2) - res;
			}
        }
        activeThreads <<= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (localId == 0)
        {
			*flipBuffFlag = 1.f - *flipBuffFlag;
        }
    }
    
    if (*flipBuffFlag > 0.f)
	{
		data0 = localBuff1[localId];
		data1 = localBuff1[localId + localSize];
	}
	else
	{
		data0 = localBuff[localId];
		data1 = localBuff[localId + localSize];
	}
    
	inBuff[buffOffset1 + localId] = data0;
	inBuff[buffOffset1 + localId + localSize] = data1;
    
}


//
// This kernel is used to transpose a matrix
//
__kernel void Mat_Transpose_kernel(__global float* inBuff, __global float* outBuff, 
								   __local float* localBuff, int width, int height)
{
	uint globalIdX = get_global_id(0);
	uint globalIdY = get_global_id(1);
	uint localIdX = get_local_id(0);
	uint localIdY = get_local_id(1);
	uint groupIdX = get_group_id(0);
	uint groupIdY = get_group_id(1);
	uint locSizeX = get_local_size(0);
	uint locSizeY = get_local_size(1);
	
	uint inIdx = (groupIdY*locSizeY+localIdY)*width + groupIdX*locSizeX + localIdX;
	localBuff[localIdY*locSizeX + localIdX] = inBuff[inIdx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	uint outIdx = (groupIdX*locSizeX+localIdY)*height + groupIdY*locSizeY + localIdX;
	outBuff[outIdx] = localBuff[localIdX*locSizeX + localIdY];
}


//
// This kernel is used to apply hard threshold on the values
//
__kernel void Mat_HT_Threshold_kernel(__global float* inBuff, __global float* outBuff, float thresh)
{
	uint globalId = get_global_id(0);

	float inVal = inBuff[globalId];
	outBuff[globalId] = (fabs(inVal) > thresh) * inVal;
}


//
// This kernel is used to apply soft threshold on the values
//
__kernel void Mat_ST_Threshold_kernel(__global float* inBuff, __global float* outBuff, float thresh)
{
	uint globalId = get_global_id(0);

	float inVal = inBuff[globalId];
	float res = fabs(inVal) - thresh;
	res = (res + fabs(res)) * 0.5f;
	outBuff[globalId] = copysign(res, inVal);
}
//-----------------------------------------------------------------------------------------
