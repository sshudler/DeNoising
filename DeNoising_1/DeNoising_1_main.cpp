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

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <CL/cl.h>

#include "Utils.h"
#include "NoiseCleaner.h"


//#define DEF_IMG_NAME	"H1N1_pig_wp_1.jpg"
//#define DEF_IMG_NAME	"test1.jpg"
#define DEF_IMG_NAME	"test2.jpg"
#define IMG_NAME_1		"test4.jpg"

enum paramType {
	PROGNAME = 0,
	INPUTIMAGENAME,
	OUTPUTIMAGENAME,
	PARAMCNT
};



int main(int argc, char *argv[])
{
	// Initialize opencv - not strictly necessary under windows
    cvInitSystem(argc, argv);

	// Load command line parameters if they exist

	const char *in = DEF_IMG_NAME;
	if (argc > INPUTIMAGENAME)
		in = argv[INPUTIMAGENAME];

	const char *out = NULL;
	if (argc > OUTPUTIMAGENAME)
		out = argv[OUTPUTIMAGENAME];

	// ----------------
	// Load input Image
	// ----------------

	std::cout << "Going to try and load image: " << in << std::endl;

	// CV_LOAD_IMAGE_GRAYSCALE should force the image to load as grayscale
    IplImage* img = cvLoadImage(in, CV_LOAD_IMAGE_GRAYSCALE);
	if (!img)
	{
		std::cerr << "Failed to load image: " << in << std::endl;
		return -1;
    }

	std::cout << "width: " << img->width << " height: " << img->height <<
		" nChunnels: " << img->nChannels << " depth: " << img->depth << std::endl;

	// For simplicity the kernel is designed for 8 bits gray scale image
	if (img->depth != 8)
	{
		std::cerr << "Unsupported depth: " << img->depth << " only support 8bits" << std::endl;
		return -1;
	}

	if (img->nChannels != 1)
	{
		std::cerr << "Unsupported depth: " << img->depth << " only support single channel" << std::endl;
		return -1;
	}

	// -------------------
	// Display Input Image
	// -------------------

	cvNamedWindow("Input", 0);
	cvResizeWindow("Input", img->width, img->height);
	cvMoveWindow("Input", 0, 50);

	cvShowImage("Input", img);

	// --------------------
	// Prepare Output Image
	// --------------------

	IplImage* oimg = cvCreateImage(cvSize(img->width, img->height), img->depth, img->nChannels);

	cvNamedWindow("Output", 0);
	cvResizeWindow("Output", oimg->width, oimg->height);
	cvMoveWindow("Output", img->width + 10, 50);

	IplImage* img_1 = cvLoadImage(IMG_NAME_1, CV_LOAD_IMAGE_GRAYSCALE);
	cvNamedWindow("Input1", 0);
	cvResizeWindow("Input1", img_1->width, img_1->height);
	cvMoveWindow("Input1", 0, img->height + 50);
	cvShowImage("Input1", img_1);


	IplImage* oimg_1 = cvCreateImage(cvSize(img_1->width, img_1->height), img_1->depth, img_1->nChannels);
	cvNamedWindow("Output1", 0);
	cvResizeWindow("Output1", img_1->width, img_1->height);
	cvMoveWindow("Output1", img_1->width + 10, oimg->height + 50);

	// give a delay to show the windows
	//cvWaitKey(500);

    // -----------------    
    // Perform Algorithm
    // -----------------

	CNoiseCleaner noiseCleaner;
	bool res = false;//noiseCleaner.PerformSelfTest();

	std::cout << "Initialized" << std::endl;
	
	int err = noiseCleaner.CleanNoise((unsigned char *)img->imageData, (unsigned char *)oimg->imageData, img->width, img->height, 0.12f, true);

	if (err)
	{
		std::cerr << "Kernel failed, exiting (error: " << err << ")" << std::endl;
		return -1;
	}
	

    // ---------------------
    // Show the output image
    // ---------------------

	cvShowImage("Output", oimg);

    // -----------------
    // Save output image
    // -----------------

	// Note that argc is 1 based and OUTPUTIMAGENAME is 0 based
	if (out)
	{
		std::cout << "Going to write file: " << out << std::endl;
		cvSaveImage(out, oimg);
	}


	

	err = noiseCleaner.CleanNoise((unsigned char *)img_1->imageData, (unsigned char *)oimg_1->imageData, img_1->width, img_1->height, 0.2f, true);
	if (err)
	{
		std::cerr << "Kernel failed, exiting (error: " << err << ")" << std::endl;
		return -1;
	}

	cvShowImage("Output1", oimg_1);

    // -----------------------------
    // Wait for user command to quit
    // -----------------------------

	while ((char)cvWaitKey() <= 0);

	// -------
	// Cleanup
	// -------

	cvDestroyWindow("Input");
	cvDestroyWindow("Output");
	cvDestroyWindow("Input1");
	cvDestroyWindow("Output1");

	cvReleaseImage(&img);
	cvReleaseImage(&oimg);
	cvReleaseImage(&img_1);
	cvReleaseImage(&oimg_1);

	return 0;
}
