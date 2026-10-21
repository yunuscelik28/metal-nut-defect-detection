#include "image_processor.hpp"

using namespace std;
using namespace cv;

NutMask segmentNut(const Mat &image) {
  NutMask nut;

  Mat grayImage, blurImage;
  cvtColor(image, grayImage, COLOR_BGR2GRAY);
  GaussianBlur(grayImage, blurImage, Size(5, 5), 0);

  // background is dark and the nut is bright, so Otsu works well here.
  // First I tried the saturation channel (from my tomato test), but on
  // metal_nut the background noise had more saturation than the nut.
  Mat binaryImage;
  double otsuThreshold =
      threshold(blurImage, binaryImage, 0, 255, THRESH_BINARY + THRESH_OTSU);

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(binaryImage, contours, hierarchy, RETR_CCOMP,
               CHAIN_APPROX_SIMPLE);

  // biggest contour without a parent (hierarchy[i][3] == -1) is the nut
  int outerIndex = -1;
  double largestArea = 0.0;

  for (int i = 0; i < static_cast<int>(contours.size()); ++i) {
    if (hierarchy[i][3] != -1) {
      continue;
    }

    double area = contourArea(contours[i]);

    if (area > largestArea) {
      largestArea = area;
      outerIndex = i;
    }
  }

  if (outerIndex == -1) {
    return nut;
  }

  // only the outer contour is filled, so the small dark spots on the
  // surface (inner contours) disappear from the mask
  Mat outerMask = Mat::zeros(binaryImage.size(), CV_8UC1);
  drawContours(outerMask, contours, outerIndex, Scalar(255), FILLED);

  /*
   * Finding the hole:
   * - distanceTransform: for every dark pixel, distance to the closest
   *   bright pixel
   * - the pixel in the middle of the hole has the biggest distance, and
   *   that distance is the hole radius
   * - we only search near the nut center, because the background is dark
   *   too and would win otherwise
   * Before, I took the biggest inner contour as the hole. On some images
   * the dark surface texture joined the hole and it became a huge blob
   * (bent/003), this way doesn't have that problem.
   */
  Mat distImage;
  distanceTransform(~binaryImage, distImage, DIST_L2, 3);

  Rect box = boundingRect(contours[outerIndex]);
  Point nutCenter(box.x + box.width / 2, box.y + box.height / 2);

  Mat searchMask = Mat::zeros(binaryImage.size(), CV_8UC1);
  circle(searchMask, nutCenter, 60, Scalar(255), FILLED);

  double holeRadius = 0.0;
  Point holeCenter;
  minMaxLoc(distImage, nullptr, &holeRadius, nullptr, &holeCenter, searchMask);

  nut.blurImage = blurImage;
  nut.otsuThreshold = otsuThreshold;
  nut.contour = contours[outerIndex];
  nut.outerMask = outerMask;
  nut.holeCenter = holeCenter;
  nut.holeRadius = holeRadius;
  nut.found = true;
  return nut;
}
