/**
 * @file main.cpp
 * @brief Inspects metal nuts for shape and color defects.
 *
 * Segments the nut, measures its perimeter, checks a surrounding
 * ring for gray pixels, and detects abnormal surface colors.
 * Produces measurements, a defect decision, and a comparison image.
 */

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

struct InspectionResult {
  double perimeter = 0.0;
  int ringPixels = 0;
  int colorPixels = 0;

  bool processingOk = false;
  bool isDefective = false;
};

InspectionResult inspectNut(const Mat &image, Mat &comparison) {
  InspectionResult result;

  Mat grayImage, blurImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);
  GaussianBlur(grayImage, blurImage, Size(5, 5), 0);

  Mat binaryImage;
  double otsuThreshold =
      threshold(blurImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);

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
    return result;
  }

  result.perimeter = arcLength(contours[outerIndex], true);

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

  result.ringPixels = countNonZero(grayPixels);

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

  Mat contourOutput = outerMask.clone();
  circle(contourOutput, holeCenter, holeRadius, Scalar(0), FILLED);

  Mat labImage;
  cvtColor(image, labImage, COLOR_BGR2Lab);

  Mat colorMask;
  colorMask = contourOutput.clone();
  circle(colorMask, holeCenter, holeRadius + 30, Scalar(0), FILLED);

  int aMin = 110, bMin = 115;
  int aMax = 134, bMax = 145;
  Mat normalColor;
  inRange(labImage, Scalar(0, aMin, bMin), Scalar(255, aMax, bMax),
          normalColor);
  Mat abnormalColor = ~normalColor;
  Mat colorDefects;
  bitwise_and(abnormalColor, colorMask, colorDefects);

  Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
  morphologyEx(colorDefects, colorDefects, MORPH_OPEN, kernel);

  result.colorPixels = countNonZero(colorDefects);

  // Limits come from the 242 good images: max perimeter 2261,
  // 0 gray pixels in the ring, at most 634 color pixels.
  result.isDefective = result.perimeter > 2300 || result.ringPixels > 100 ||
                       result.colorPixels > 700;

  // Red: gray pixels in the ring, blue: color defects, green: hole.
  Mat out = image.clone();
  out.setTo(Scalar(0, 0, 255), grayPixels);
  out.setTo(Scalar(255, 0, 0), colorDefects);
  circle(out, holeCenter, holeRadius, Scalar(0, 255, 0), 2);

  Mat maskBgr;
  cvtColor(contourOutput, maskBgr, COLOR_GRAY2BGR);

  hconcat(out, maskBgr, comparison);

  result.processingOk = true;
  return result;
}

int runFolder(const string &folder) {
  vector<string> files;
  glob(folder + "/*.png", files, false);

  if (files.empty()) {
    cerr << "Error: folder is empty..." << "\n";
    return -1;
  }

  ofstream csv("results.csv");

  if (!csv.is_open()) {
    cerr << "Error: results.csv couldn't open..." << "\n";
    return -1;
  }

  csv << "file,perimeter,ringPixels,colorPixels,result" << "\n";
  csv << fixed << setprecision(2);

  int defectCount = 0;
  int okCount = 0;

  for (int i = 0; i < static_cast<int>(files.size()); ++i) {
    string path = files[i];

    Mat image = imread(path);

    if (image.empty()) {
      cerr << "Error: Image file not found at " << path << "\n";
      continue;
    }

    Mat comparison;
    InspectionResult inspection = inspectNut(image, comparison);

    if (!inspection.processingOk) {
      cerr << "Error: nut not found" << "\n";
      csv << path << ",0,0,0,ERROR" << "\n";
      continue;
    }

    csv << path << "," << inspection.perimeter << "," << inspection.ringPixels
        << "," << inspection.colorPixels << ","
        << (inspection.isDefective ? "DEFECT" : "OK") << "\n";

    if (inspection.isDefective) {
      ++defectCount;
    } else {
      ++okCount;
    }
  }

  csv.close();

  if (!csv) {
    cerr << "An error occurred while writing the CSV.\n";
    return -1;
  }

  cout << defectCount << " DEFECT / " << okCount << " OK  " << "\n";

  cout << "The results have been written to the results.csv file..\n";

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: ./mtest <image_or_folder>\n";
    return -1;
  }

  // Input image file path
  string path = argv[1];

  if (filesystem::is_directory(path)) {
    return runFolder(path);
  }

  // Load the target image from disk
  Mat image = imread(path);

  // Check if the image was successfully loaded
  if (image.empty()) {
    cerr << "Error: Image file not found at " << path << "\n";
    return -1;
  }

  Mat comparison;

  InspectionResult inspection = inspectNut(image, comparison);

  if (!inspection.processingOk) {
    cerr << "Error: nut not found" << "\n";
    return -1;
  }

  if (comparison.empty()) {
    cerr << "Comparison image couldn't be created!" << "\n";
    return -1;
  }

  cout << "Perimeter: " << inspection.perimeter << "\n";
  cout << "Ring pixels: " << inspection.ringPixels << "\n";
  cout << "Color pixels: " << inspection.colorPixels << "\n";
  cout << "Result: " << (inspection.isDefective ? "DEFECT" : "OK") << "\n";

  if (!imwrite("nut_result.png", comparison)) {
    cerr << "Could not save the result.\n";
    return -1;
  }

  imshow("Result", comparison);
  waitKey(0);
  return 0;
}
