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

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: ./mtest <image_path>\n";
    return -1;
  }

  // Input image file path
  string path = argv[1];

  // Load the target image from disk
  Mat image = imread(path);

  // Check if the image was successfully loaded
  if (image.empty()) {
    cerr << "Error: Image file not found at " << path << "\n";
    return -1;
  }

  Mat grayImage, blurImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);
  GaussianBlur(grayImage, blurImage, Size(5, 5), 0);

  Mat binaryImage;
  double otsuThreshold =
      threshold(blurImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);
  cout << "Otsu threshold: " << otsuThreshold << "\n";

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(binaryImage, contours, hierarchy, RETR_CCOMP,
               CHAIN_APPROX_SIMPLE);

  int outerIndex = -1;
  double largestOA = 0.0;

  for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
    if (hierarchy[i][3] != -1) {
      continue;
    }

    double area = contourArea(contours[i]);

    if (area > largestOA) {
      largestOA = area;
      outerIndex = i;
    }
  }

  if (outerIndex == -1) {
    cerr << "No outer contour found.\n";
    return -1;
  }

  vector<Point> hull;
  convexHull(contours[outerIndex], hull);

  double hullArea = contourArea(hull);
  double solidity = 0.0;
  double perimeter = 0.0;

  if (hullArea > 0) {
    solidity = largestOA / hullArea;
  }

  perimeter = arcLength(contours[outerIndex], true);

  cout << "Area: " << largestOA << "\n";
  cout << "Solidity: " << solidity << "\n";
  cout << "Perimeter: " << perimeter << "\n";

  Mat outerMask = Mat::zeros(binaryImage.size(), CV_8UC1);

  drawContours(outerMask, contours, outerIndex, Scalar(255), FILLED);

  Mat ringKernel = getStructuringElement(MORPH_ELLIPSE, Size(71, 71));

  Mat expandedMask;
  dilate(outerMask, expandedMask, ringKernel);

  Mat ringMask;
  subtract(expandedMask, outerMask, ringMask);

  Mat contourOutput = outerMask.clone();

  int holeIndex = -1;
  double largestHA = 0.0;

  for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
    if (hierarchy[i][3] != outerIndex) {
      continue;
    }

    double area = contourArea(contours[i]);

    if (area > largestHA) {
      largestHA = area;
      holeIndex = i;
    }
  }

  if (holeIndex != -1) {
    drawContours(contourOutput, contours, holeIndex, Scalar(0), FILLED);
  } else {
    cerr << "No hole found in the selected contour.\n";
  }

  Mat maskBgr;
  cvtColor(contourOutput, maskBgr, COLOR_GRAY2BGR);

  Mat comparison;
  hconcat(image, maskBgr, comparison);

  imshow("1 - Deliksiz maske", outerMask);
  imshow("2 - Genisletilmis maske", expandedMask);
  imshow("3 - Dis halka", ringMask);

  if (!imwrite("nut_result.png", comparison)) {
    cerr << "Could not save the result.\n";
  }
  waitKey(0);
  return 0;
}
