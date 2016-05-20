#include "Segmentation.hpp"

int main(int argc, char const *argv[])
{
	
  if (argc != 2) {
    cout << "Usage: DisplayImage.out <Image_Path>\n";
    return -1;
  }

  ImgSegmenter segm(cv::imread(argv[1], CV_16UC1));

  segm.estimateNormals();
  segm.printNormals();
  segm.detectNormalEdges();
  segm.colorRegions();

  cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
  cv::imshow("Original Image", segm.image);

  cv::namedWindow("Surface Normals Image", CV_WINDOW_AUTOSIZE );
  cv::imshow("Surface Normals Image", segm.norm_img);

  cv::namedWindow("Surface Normals Image Mapped to RGB", CV_WINDOW_AUTOSIZE );
  cv::imshow("Surface Normals Image Mapped to RGB", segm.norm_color_img);

  cv::namedWindow("Surface Edges", CV_WINDOW_AUTOSIZE );
  cv::imshow("Surface Edges", segm.norm_edge_img);
  
  cv::namedWindow("Surface Binary Edges", CV_WINDOW_AUTOSIZE );
  cv::imshow("Surface Binary Edges", segm.norm_bin_edge_img);

  cv::namedWindow("Colored Regions", CV_WINDOW_AUTOSIZE );
  cv::imshow("Colored Regions", segm.colored_img);

  cv::waitKey(0);

	return 0;
}