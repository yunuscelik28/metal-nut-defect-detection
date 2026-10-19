/**
 * @file main.cpp
 * @brief Metal nut segmentation and bent-tab detection (MVTec AD metal_nut).
 *
 * Segments the nut with grayscale Otsu, measures its outer contour, finds
 * the center hole as the largest dark circle near the centroid, and flags a
 * bent tab when the silhouette perimeter is too long or gray pixels show up
 * in a band outside the silhouette.
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

/*
 * Thresholds measured on all 220 train/good + 22 test/good images:
 * - Good perimeters are 2154..2261 px, so 2300 leaves a small margin.
 * - Good nuts never have gray pixels farther than ~15 px from their edge
 *   (max 0 in the band), bent ones have 1000+ when the perimeter misses them.
 */
const double kPerimeterLimit = 2300.0;
const int kBandInnerKernel = 31;  // band starts ~15 px outside the silhouette
const int kBandOuterKernel = 101; // and ends ~50 px outside it
const double kBackgroundMargin = 10.0;
const int kBandPixelLimit = 100;
const int kHoleSearchRadius = 60;

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

  /*
   * Background level:
   * - farMask: everything well away from the nut (pure background).
   * - backgroundLevel: mean brightness there, used as the lower gray bound.
   */
  Mat farMask;
  dilate(outerMask, farMask,
         getStructuringElement(MORPH_ELLIPSE, Size(151, 151)));
  double backgroundLevel = mean(blurImage, ~farMask)[0];

  /*
   * Outer band check (bent tabs):
   * A bent tab is darker than the nut, so Otsu drops it from the silhouette,
   * but it is still brighter than the background. The first ~15 px around a
   * good nut are a soft blurry edge, so the band starts beyond that.
   */
  Mat bandOuter, bandInner, bandMask;
  dilate(outerMask, bandOuter,
         getStructuringElement(MORPH_ELLIPSE,
                               Size(kBandOuterKernel, kBandOuterKernel)));
  dilate(outerMask, bandInner,
         getStructuringElement(MORPH_ELLIPSE,
                               Size(kBandInnerKernel, kBandInnerKernel)));
  subtract(bandOuter, bandInner, bandMask);

  Mat grayPixels;
  inRange(blurImage, backgroundLevel + kBackgroundMargin, otsuThreshold,
          grayPixels);
  bitwise_and(grayPixels, bandMask, grayPixels);
  int bandPixels = countNonZero(grayPixels);

  cout << "Background level: " << backgroundLevel << "\n";
  cout << "Band gray pixels: " << bandPixels << "\n";

  /*
   * Center hole:
   * The contour hierarchy is unreliable here (dark surface texture can merge
   * with the hole), so the hole is the largest dark circle near the centroid.
   * The convex hull is used as the search area so a stain that connects the
   * hole to the background does not hide it.
   */
  Moments nutMoments = moments(contours[outerIndex]);
  Point2d centroid(nutMoments.m10 / nutMoments.m00,
                   nutMoments.m01 / nutMoments.m00);

  Mat hullMask = Mat::zeros(binaryImage.size(), CV_8UC1);
  fillConvexPoly(hullMask, hull, Scalar(255));

  Mat darkInside;
  bitwise_and(~binaryImage, hullMask, darkInside);
  morphologyEx(darkInside, darkInside, MORPH_OPEN,
               getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

  Mat distToBright;
  distanceTransform(darkInside, distToBright, DIST_L2, 5);

  Mat searchArea = Mat::zeros(binaryImage.size(), CV_8UC1);
  circle(searchArea, centroid, kHoleSearchRadius, Scalar(255), FILLED);

  double holeRadius = 0.0;
  Point holeCenter;
  minMaxLoc(distToBright, nullptr, &holeRadius, nullptr, &holeCenter,
            searchArea);

  cout << "Hole radius: " << holeRadius << "\n";

  Mat contourOutput = outerMask.clone();
  circle(contourOutput, holeCenter, cvRound(holeRadius), Scalar(0), FILLED);

  /*
   * Decision:
   * Either signal alone is enough to flag the nut.
   */
  bool longPerimeter = perimeter > kPerimeterLimit;
  bool grayOutside = bandPixels > kBandPixelLimit;
  bool isDefective = longPerimeter || grayOutside;

  cout << "Result: " << (isDefective ? "DEFECT" : "OK");
  if (longPerimeter) {
    cout << " [perimeter]";
  }
  if (grayOutside) {
    cout << " [gray band]";
  }
  cout << "\n";

  // Overlay: band hits in red, hole circle in green, verdict in the corner.
  Mat overlay = image.clone();
  overlay.setTo(Scalar(0, 0, 255), grayPixels);
  circle(overlay, holeCenter, cvRound(holeRadius), Scalar(0, 255, 0), 2);
  putText(overlay, isDefective ? "DEFECT" : "OK", Point(20, 50),
          FONT_HERSHEY_SIMPLEX, 1.5,
          isDefective ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 3);

  Mat maskBgr;
  cvtColor(contourOutput, maskBgr, COLOR_GRAY2BGR);

  Mat comparison;
  hconcat(overlay, maskBgr, comparison);

  if (!imwrite("nut_result.png", comparison)) {
    cerr << "Could not save the result.\n";
  }

  // Pass --no-gui as the second argument to skip the window (batch runs).
  bool showWindow = !(argc > 2 && string(argv[2]) == "--no-gui");
  if (showWindow) {
    imshow("Overlay | Mask", comparison);
    waitKey(0);
  }
  return 0;
}
