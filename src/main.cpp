#include <functional>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main() {
  // Input image file path
  string path = "../data/metal_nut/train/good/000.png";

  // Load the target image from disk
  Mat image = imread(path);

  // Check if the image was successfully loaded
  if (image.empty()) {
    cout << "Error: Image file not found at " << path << endl;
    cin.get();
    return -1;
  }

  Mat grayImage, blurImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);
  GaussianBlur(grayImage, blurImage, Size(5, 5), 0);

  Mat hsvImage;
  cvtColor(image, hsvImage, COLOR_BGR2HSV);

  vector<Mat> hsvChannels;
  split(hsvImage, hsvChannels);

  Mat saturationBlr;
  GaussianBlur(hsvChannels[1], saturationBlr, Size(5, 5), 0);

  Mat binaryImage;
  threshold(blurImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  Mat contourOutput = Mat::zeros(binaryImage.size(), CV_8UC1);
  findContours(binaryImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
  drawContours(contourOutput, contours, -1, Scalar(255), FILLED);

  Mat cleanImage;
  Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
  morphologyEx(contourOutput, cleanImage, MORPH_OPEN, kernel);

  // Display the final output image
  // imshow("Test", out);
  // Wait for any key press before closing
  imshow("Test", binaryImage);
  waitKey(0);
  return 0;
}
