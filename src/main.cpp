/**
 * @file main.cpp
 * @brief Finds the metal nut in the image and checks it for bent tabs.
 */

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

  // Ring around the nut. The first ~15 px are the blurry edge of the nut,
  // so the ring starts after that (small kernel) and ends ~50 px out.
  Mat smallKernel = getStructuringElement(MORPH_ELLIPSE, Size(31, 31));
  Mat bigKernel = getStructuringElement(MORPH_ELLIPSE, Size(101, 101));

  Mat smallExpanded, bigExpanded;
  dilate(outerMask, smallExpanded, smallKernel);
  dilate(outerMask, bigExpanded, bigKernel);

  Mat ringMask;
  subtract(bigExpanded, smallExpanded, ringMask);

  // A bent tab is darker than the nut but brighter than the background.
  // Background is about 25, so gray means between 35 and the Otsu value.
  Mat grayPixels;
  inRange(blurImage, 35, otsuThreshold, grayPixels);
  bitwise_and(grayPixels, ringMask, grayPixels);

  int ringPixels = countNonZero(grayPixels);
  cout << "Gray pixels in ring: " << ringPixels << "\n";

  // Center hole: find the dark pixel that is farthest from any bright pixel.
  // Its distance is the hole radius. Only search near the nut center.
  Mat distImage;
  distanceTransform(~binaryImage, distImage, DIST_L2, 3);

  Rect box = boundingRect(contours[outerIndex]);
  Point nutCenter(box.x + box.width / 2, box.y + box.height / 2);

  Mat searchMask = Mat::zeros(binaryImage.size(), CV_8UC1);
  circle(searchMask, nutCenter, 60, Scalar(255), FILLED);

  double holeRadius = 0.0;
  Point holeCenter;
  minMaxLoc(distImage, nullptr, &holeRadius, nullptr, &holeCenter, searchMask);
  cout << "Hole radius: " << holeRadius << "\n";

  Mat contourOutput = outerMask.clone();
  circle(contourOutput, holeCenter, holeRadius, Scalar(0), FILLED);

  // Limits come from the good images: max perimeter was 2261 and good nuts
  // had 0 gray pixels in the ring.
  bool isDefective = perimeter > 2300 || ringPixels > 100;
  cout << "Result: " << (isDefective ? "DEFECT" : "OK") << "\n";

  // Red: gray pixels found in the ring, green: hole.
  Mat result = image.clone();
  result.setTo(Scalar(0, 0, 255), grayPixels);
  circle(result, holeCenter, holeRadius, Scalar(0, 255, 0), 2);

  Mat maskBgr;
  cvtColor(contourOutput, maskBgr, COLOR_GRAY2BGR);

  Mat comparison;
  hconcat(result, maskBgr, comparison);

  if (!imwrite("nut_result.png", comparison)) {
    cerr << "Could not save the result.\n";
  }

  imshow("Result | Mask", comparison);
  waitKey(0);
  return 0;
}
