#pragma once
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// Everything segmentNut() finds about the nut.
// defect_detector uses these, so it doesn't have to find the nut again.
struct NutMask {
  cv::Mat blurImage;              // blurred gray image
  std::vector<cv::Point> contour; // outer contour of the nut
  cv::Mat outerMask;              // nut white, rest black (hole is white too)
  cv::Point holeCenter;
  double otsuThreshold = 0.0;
  double holeRadius = 0.0;

  bool found = false; // stays false if there is no nut in the image
};

/*
 * segmentNut
 * - Otsu on the gray image: nut white, background black
 * - biggest outer contour = the nut
 * - hole = biggest dark circle near the middle of the nut
 */
NutMask segmentNut(const cv::Mat &image);
