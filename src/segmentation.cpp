#include "segmentation.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "usage: DisplayImage.out <Image_Path>\n";
        return -1;
    }
    
    image = imread(argv[1], CV_8UC1);

    if (!image.data) {
        cout << "No image data \n";
        return -1;
    }
       
    median_img = image.clone();
    norm_img = median_img.clone();
    norm_edge_img = median_img.clone();
    norm_color_img = median_img.clone();
    
    kernel_size = 3;
    radius = 3;
    kernel_normedge = 4; 
    
    // Recursive Median Filter for Noise Elimination
    //*
    begin = clock();
    cout << "Recursive Median Filtering begin" << endl;

    recursiveMedian( median_img, kernel_size);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Recursive Median Filtering end " << endl << "Elapsed time: " << elapsed_secs << endl; 
    //*/

    // Estimation of Surface Normals
    //*
    begin = clock();
    cout << "Estimation of Surface Normals begin" << endl;

    normals.reserve( (median_img.rows - 2 * radius - 1) * (median_img.cols - 2 * radius - 1));    
    estimate_normals( median_img, radius, normals);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Estimation of Surface Normals end " << endl << "Elapsed time: " << elapsed_secs << endl; 
    //*/
   
    // Prints Normal Vector at every pixel and Map xyz to RGB
    //*
    begin = clock();
    cout << "Printing of Surface Normals begin" << endl;
    
    cvtColor(image, norm_color_img, CV_GRAY2RGB); // Make norm_color_img 3 channels
    
    Rect crop(radius, radius, norm_color_img.cols - 2 * radius, norm_color_img.rows - 2 * radius);
    norm_color_img = norm_color_img(crop);
    norm_img = norm_img(crop);
    
    print_normals( norm_img, norm_color_img, normals, radius, 10);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Printing of Surface Normals end " << endl << "Elapsed time: " << elapsed_secs << endl; 
    //*/

    // Detection of Surface Normal Edges
    //*
    begin = clock();
    cout << "Detection of Surface Normal Edges begin" << endl;

    norm_edge_img = norm_edge_img(crop);

    detect_normal_edges( norm_edge_img, normals, radius, kernel_normedge);

    Rect crop1(kernel_normedge, kernel_normedge, norm_edge_img.cols - 2 * kernel_normedge, norm_edge_img.rows - 2 * kernel_normedge);
    norm_edge_img = norm_edge_img(crop1);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Detection of Surface Normal Edges end " << endl << "Elapsed time: " << elapsed_secs << endl; 
    //*/

    //medianBlur(norm_color_img, norm_color_img, 5);
    medianBlur(norm_edge_img, norm_edge_img, 5);

    median_img = median_img(crop);
    median_img = median_img(crop1);

    //*
    begin = clock();
    cout << "Region Growing begin" << endl;
    
    Mat segment;

    cvtColor(norm_edge_img, segment, CV_GRAY2RGB);

    int numofRegions = 0;
    
    for (int i = 0; i < segment.rows; ++i) {
        for (int j = 0; j < segment.cols; ++j) {
            if (segment.at<Vec3b>(i, j)[2] == 255 &&
                segment.at<Vec3b>(i, j)[1] == 255 &&
                segment.at<Vec3b>(i, j)[0] == 255) {
                ++numofRegions;
                Point2i seed(j, i);
                int color1 = 10 * numofRegions;
                int color2 = 10 * (numofRegions + 1);
                int color3 = 10 * (numofRegions - 1);
                Scalar color(color1, color2, color3);
                floodFill(segment, seed, color);    
            }
        }
    }

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Region Growing end \n"; 
    cout << "Found " << numofRegions << " regions \n";
    cout << "Elapsed time: " << elapsed_secs << "\n"; 
    //*/

    // Shows Images
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE );
    imshow("Original Image", image);

    //namedWindow("Median Image", CV_WINDOW_AUTOSIZE );
    //imshow("Median Image", median_img);
    
    namedWindow("Surface Normals Image", CV_WINDOW_AUTOSIZE );
    imshow("Surface Normals Image", norm_img);
    
    //namedWindow("Surface Normals Image Mapped to RGB", CV_WINDOW_AUTOSIZE );
    //imshow("Surface Normals Image Mapped to RGB", norm_color_img);
    
    namedWindow("Surface Normal Edges Image", CV_WINDOW_AUTOSIZE );
    imshow("Surface Normal Edges Image", norm_edge_img);

    namedWindow("Segmented Image", CV_WINDOW_AUTOSIZE );
    imshow("Segmented Image", segment);
    
    waitKey(0);
}

void recursiveMedian (Mat& dst, const int kernel_size) {
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            std::vector<uchar> v;
            
            for (int i = - kernel_size / 2; i <= kernel_size / 2; ++i) {
                for (int j = - kernel_size / 2; j <= kernel_size / 2; ++j) {
                    // Clamp Image Boundary
                    int image_r = std::min(std::max(r + i, 0), static_cast<int>(image.rows - 1));
                    int image_c = std::min(std::max(c + j, 0), static_cast<int>(image.cols - 1));
                    // Check if Pixel is Valid
                    if ((int)dst.at<uchar>( image_r, image_c) != 0) {
                        v.push_back(dst.at<uchar>( image_r, image_c));                        
                    }
                }
            }
            // Find median of kernel window
            if (v.size() > 0) {   
                std::sort(v.begin(), v.end());
                dst.at<uchar>(r , c) = v[v.size() / 2];            
            }
        }
    }
}

void estimate_normals (const Mat& img, const int radius, 
                        std::vector<Point3f>& norm) {
    for (int i = radius; i < img.rows - radius; ++i) {
        for (int j = radius; j < img.cols - radius; ++j) {
            // Points a, b, c in the neighborhood of pixel (i, j)
            Point3f a( i + radius, j - radius, ( float )img.at<uchar>(i + radius, j - radius));
            Point3f b( i + radius, j + radius, ( float )img.at<uchar>(i + radius, j + radius));
            Point3f c( i - radius, j, ( float )img.at<uchar>(i - radius, j));
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
                float norm1 = sqrt( pow(n.x, 2) + pow(n.y, 2) + pow(n.z, 2));
                // Check if length is zero
                if (norm1 == 0) {
                    norm1 = 1;
                }
                n.x = n.x / norm1;
                n.y = n.y / norm1;
                n.z = n.z / norm1;                
            }   
            norm.push_back( n );     
        }
    }
}

void print_normals (Mat& arrowed_dst, Mat& color_dst, 
                    const std::vector<Point3f> norm, 
                    const int radius, const int kern) {
    for (int i = 0; i < arrowed_dst.rows; ++i) {
        for (int j = 0; j < arrowed_dst.cols; ++j) {
            // Calculate Index
            int index = i * arrowed_dst.cols + j;
            // Map normals x, y, z to RGB           
            color_dst.at<Vec3b>(i ,j)[0] = (255 * norm[ index ].x);
            color_dst.at<Vec3b>(i, j)[1] = (255 * norm[ index ].y);
            color_dst.at<Vec3b>(i, j)[2] = (255 * norm[ index ].z);

            // if (color_dst.at<Vec3b>(i ,j)[0] < 100)
            // {
            //     color_dst.at<Vec3b>(i ,j)[0] = 0;
            //     color_dst.at<Vec3b>(i, j)[1] = 0;
            //     color_dst.at<Vec3b>(i, j)[2] = 0;
            // }
            // else
            // {
            //     color_dst.at<Vec3b>(i ,j)[0] = 255;
            //     color_dst.at<Vec3b>(i ,j)[1] = 255;
            //     color_dst.at<Vec3b>(i ,j)[2] = 255;  
            // }

            // Print normals with step kern
            if (i % kern == 0 && j % kern == 0) {
                Point3f n = norm[ index ];
                Point p4( 25 * n.y, 25 * n.x);
                Point p5( j, i);
                Point p6 = p5 + p4;
                arrowedLine( arrowed_dst, p5, p6, Scalar(255, 255, 255));
            }
        }
    }
}

void detect_normal_edges (Mat& dst, const std::vector<Point3f> norm,
                        const int radius, const int kernel_normedge) {
    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            if (i >= kernel_normedge && 
                j >= kernel_normedge &&
                i < dst.rows - kernel_normedge &&
                j < dst.cols - kernel_normedge) {
                // Calculate Index 
                int index = i * dst.cols + j;

                Point3f n0 = norm[index];
                // Angles in the 8 directions 
                float costheta = 0;     // Min angle 
                float costhetan = 0;    // North
                float costhetas = 0;    // South
                float costhetaw = 0;    // West
                float costhetae = 0;    // East
                float costhetanw = 0;   // North-West
                float costhetane = 0;   // North-East
                float costhetasw = 0;   // South-West
                float costhetase = 0;   // South-East

                for (int count = 1; count <= kernel_normedge; ++count) {
                    // North Direction
                    Point3f n = normals[(i - count) * dst.cols + j ];
                    costhetan += n0.dot(n) / kernel_normedge;

                    // South Direction
                    Point3f s = normals[(i + count) * dst.cols + j ];
                    costhetas += n0.dot(s) / kernel_normedge;

                    // West Direction
                    Point3f w = normals[ i * dst.cols + (j - count)];
                    costhetaw += n0.dot(w) / kernel_normedge;

                    // East Direction
                    Point3f e = normals[ i * dst.cols + (j + count)];
                    costhetae += n0.dot(e) / kernel_normedge;

                    // North West Direction
                    Point3f nw = normals[(i - count) * dst.cols + (j - count)];
                    costhetanw += n0.dot(nw) / kernel_normedge;

                    // North East Direction
                    Point3f ne = normals[(i - count) * dst.cols + (j + count)];
                    costhetane += n0.dot(ne) / kernel_normedge;

                    // South West Direction
                    Point3f sw = normals[(i + count) * dst.cols + (j - count)];
                    costhetasw += n0.dot(sw) / kernel_normedge;                    

                    // South East Direction
                    Point3f se = normals[(i + count) * dst.cols + (j + count)];
                    costhetase += n0.dot(se) / kernel_normedge; 
                }
                 
                std::vector<float> thetas{costhetan, costhetas, costhetaw, 
                                        costhetae, costhetanw, costhetane, 
                                        costhetasw, costhetase};

                costheta = *std::min_element(thetas.begin(), thetas.end());
                //costheta = costheta / 255;
                //cout << " costheta " << costheta << endl; 
                //*
                // Print Binarized costhetas
                if (costheta > .93) {
                    dst.at<uchar>(i, j) = 255;
                } else {
                    dst.at<uchar>(i, j) = 0;   
                }
                //*/
                // Print costhetas
                //dst.at<uchar>( i , j ) = costheta * 255 ;
            }
        }
    }
}
