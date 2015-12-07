#ifndef SEGMENTATION_HPP_
#define SEGMENTATION_HPP_

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui_c.h"

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <initializer_list>

using std::cout;
using namespace cv;

void estimate_normals (const Mat& img, const int radius, std::vector<Point3f>& norm);

Mat image;			// Original Input Image
Mat median_img; 	// Median Blured Image

int kernel_size; 	// Median Filter Kernel
int radius; 		// Radius of Normal Estimation Triangle

clock_t begin, end;
double elapsed_secs;

std::vector<Point3f> normals; // Vector of Normals for every Point

#endif // SEGMENTATION_HPP_