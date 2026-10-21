#pragma once
#include <opencv2/opencv.hpp>

// Measurements and the final decision for one image.
struct InspectionResult {
  double perimeter = 0.0;
  int ringPixels = 0;  // gray pixels outside the nut (bent tab)
  int colorPixels = 0; // pixels with a wrong color (stain)

  bool processingOk = false; // false if no nut was found
  bool isDefective = false;
};

/*
 * inspectNut
 * - bent tab: perimeter too long, or gray pixels in a ring outside the nut
 * - color stain: Lab colors that good nuts don't have
 * - comparison: output image, result on the left and the mask on the right
 */
InspectionResult inspectNut(const cv::Mat &image, cv::Mat &comparison);
