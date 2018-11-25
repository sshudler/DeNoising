# DeNoising

The main component in this package is the CNoiseCleaner class which implements
GPU-based denoising algorithm in much the same way as WaveLab's 'ThreshWave2'
function. However instead of using the CPU it employs OpenCL and the GPU to
accelerate the algorithm.
In order to fully utilize the GPU there is a limit on the size of the images
the algorithm can handle. Maximal possible value for the width and height is
1024 pixels.
The algorithm has 5 stages:
1. Simultenous forward Haar transform on all of the rows. Each row is processed
   by a different work-group on the GPU, such that the total number of workgroups
   is the number of rows (the height of the image). The name of the kernel which
   is invoked in this stage is `FWT_kernel`. The way this kernel operates is the
   reason for the limit (1024 pixels) on the image size. The number of work-items
   in a work-group must be half the number of pixels in a row for the kernel to
   work properly. Increasing the size of the image to 2048 and above would require
   work-groups with at least 1024 work-items and current GPUs don't have such
   capability yet.
2. Transposing the image such that the columns of the original image become the
   rows in the transposed one. The name of the kernel which is invoked in this
   stage is `Mat_Transpose_kernel`.
3. Simultenous forward Haar transform on all of the rows in the transposed
   image. This stage is exactly as the first one, but it actually runs on the
   columns of the original image. This is the reason why the limit of 1024 pixels
   is true for both the width and the height of the image.
4. The thresholding step which either invokes `Mat_HT_Threshold_kernel` for hard-
   thresholding or `Mat_ST_Threshold_kernel` for soft-thresholding depending on
   the type of thresholding requested by the user.   
5. Simultenous inverse Haar transform on all the rows in the transposed image,
   which means the columns of the original matrix. The inverse transform works in
   much the same way as the forward transform but the kernel, which is called
   `IWT_kernel`, performs a sort of an expansion rather than a reduction.
6. Transposing the image back into its original form using the same
   `Mat_Transpose_kernel` kernel.
7. Simultenous inverse Haar transform on all the rows in the image. Exactly as
   the fifth stage but it runs on the rows of the image rather than on its
   columns.
   
See NoiseCleaner.h for detailed API.


List of files for DeNoising package:

DeNoising.sln - Visual Studio 2008 solution file.
DeNoising.suo - Visual Studio user options.
DeNoising_1\
	DeNoising_1.vcproj - Visual Studio 2008 project file.
	DeNoising_1_main.cpp - A simple test program which tests the functionality
			       of the GPU-based denoising algorithm. It uses OpenCV
			       to read and display the image files.
			       This program also serves as an example for the usage of
			       the CNoiseCleaner class which encapsulates the denoising
			       algorithm.
	HWT_kernels.cl - Contains OpenCL kernels for various stages of the denoising
			 algorithm. This file must be present in the current directory
			 of the application. The kernels are compiled dynamically during
			 runtime, without them the algorithm will not work.
	Makefile - A makefile for compiling the test application in Linux. Serves as an
		   example and can be further extended as needed.
	NoiseCleaner.cpp - Implementation of the CNoiseCleaner class. See comments in
			   the file for further details.
	NoiseCleaner.h - Header for CNoiseCleaner class.
	*.dat - The files that start with 'signal' contain 1D signal in various sizes, and
		the ones that start with 'regression' contain the corresponding wavelet
		coefficients. These file are used to test the forward and inverse Haar transform
		methods in CNoiseCleaner class. This files are not essential and are needed only
		when CNoiseCleaner::PerformSelfTest() method is activated.
	*.jpg - Test images for the test program in DeNoising_1_main.cpp.
	Utils.cpp - Implementations of various auxiliary functions for working with files and OpenCL.
	Utils.h - Header file for various auxiliary functions for working with files and OpenCL.
