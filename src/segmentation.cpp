#include "segmentation.hpp"

int main(int argc, char** argv) {
    
    if (argc != 2) {
        cout << "Usage: DisplayImage.out <Image_Path>\n";
        return -1;
    }
    
    image = imread(argv[1], CV_8UC1);

    if (!image.data) {
        cout << "No image data \n";
        return -1;
    }
    
    kernel_size = 3;
    medianBlur(image, median_img, kernel_size); // Median Filter for Noise Reduction

    // Estimation of Surface Normals
    //*
    begin = clock();
    cout << "Estimation of Surface Normals begin \n";

    normals.reserve((median_img.rows - 2 * radius - 1) * (median_img.cols - 2 * radius - 1));    
    estimate_normals( median_img, radius, normals);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Estimation of Surface Normals end \n Elapsed time: " << elapsed_secs << "\n"; 
    //*/


    // Shows Images
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE );
    imshow("Original Image", image);
    
    namedWindow("Median Image", CV_WINDOW_AUTOSIZE );
    imshow("Median Image", median_img);

    waitKey(0);   
}


void estimate_normals (const Mat& img, const int radius, 
                        std::vector<Point3f>& normals) {
    for (int i = radius; i < img.rows - radius; ++i) {
        for (int j = radius; j < img.cols - radius; ++j) {
            // Points a, b, c in the neighborhood of pixel (i, j)
            Point3f a( i + radius, j - radius, (float)img.at<uchar>(i + radius, j - radius));
            Point3f b( i + radius, j + radius, (float)img.at<uchar>(i + radius, j + radius));
            Point3f c( i - radius, j, (float)img.at<uchar>(i - radius, j));
            Point3f n;
            // Check if pixel is not a valid measurment
            if (a.z == 0 || b.z == 0 || c.z == 0) {   
                n = Point3f(0, 0, 0);
            } else {
                // Vectors for normal estimation
                Point3f v1 = b - a;
                Point3f v2 = c - a;
                // Normal is the cross product of v1 and v2
                n = v1.cross(v2);
                // Normalize vector
                float norm = sqrt( pow(n.x, 2) + pow(n.y, 2) + pow(n.z, 2));
                // Check if length is zero
                if (norm == 0) {
                    norm = 1;
                }
                n.x = n.x / norm;
                n.y = n.y / norm;
                n.z = n.z / norm;                
            }   
            normals.push_back( n );     
        }
    }
}