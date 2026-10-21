#include "defect_detector.hpp"
#include "image_processor.hpp"

using namespace std;
using namespace cv;

InspectionResult inspectNut(const Mat &image, Mat &comparison) {
  InspectionResult result;

  NutMask nut = segmentNut(image);
  if (!nut.found) {
    return result;
  }

  // a bent tab can make the outline longer.
  // I also tried solidity, it looked good on 5 images but on all 220 good
  // images the range was too wide, so I dropped it.
  result.perimeter = arcLength(nut.contour, true);

  /*
   * Bent tab check:
   * The bent part of the tab is darker than the nut, so Otsu cuts it out
   * of the mask and the outline looks almost normal (bent/001, bent/003).
   * But it is still brighter than the background, so we look for gray
   * pixels in a ring outside the nut.
   * - first ~15 px around a good nut are just its blurry edge, so the ring
   *   starts after that (small kernel) and ends ~50 px out (big kernel)
   * - background is about 25, so "gray" = between 35 and the Otsu value
   */
  Mat smallKernel = getStructuringElement(MORPH_ELLIPSE, Size(31, 31));
  Mat bigKernel = getStructuringElement(MORPH_ELLIPSE, Size(101, 101));

  Mat smallExpanded, bigExpanded;
  dilate(nut.outerMask, smallExpanded, smallKernel);
  dilate(nut.outerMask, bigExpanded, bigKernel);

  Mat ringMask;
  subtract(bigExpanded, smallExpanded, ringMask);

  Mat grayPixels;
  inRange(nut.blurImage, 35, nut.otsuThreshold, grayPixels);
  bitwise_and(grayPixels, ringMask, grayPixels);

  result.ringPixels = countNonZero(grayPixels);

  // nut mask without the hole, also shown on the right side of the output
  Mat maskNoHole = nut.outerMask.clone();
  circle(maskNoHole, nut.holeCenter, nut.holeRadius, Scalar(0), FILLED);

  /*
   * Color stain check (Lab):
   * - a: green <-> red/pink, b: blue <-> yellow, 128 = neutral
   * - good nuts are yellow-green: a mean ~122 std ~4, b mean ~130 std ~5
   * - limits = mean +- 3 std
   * - L (brightness) is not checked, it changes too much between images
   * Pink and blue stains work. Black stains don't, they only change L.
   */
  Mat labImage;
  cvtColor(image, labImage, COLOR_BGR2Lab);

  // the dark groove around the hole is normal, so skip it too (+30 px)
  Mat colorMask = maskNoHole.clone();
  circle(colorMask, nut.holeCenter, nut.holeRadius + 30, Scalar(0), FILLED);

  int aMin = 110, bMin = 115;
  int aMax = 134, bMax = 145;
  Mat normalColor;
  inRange(labImage, Scalar(0, aMin, bMin), Scalar(255, aMax, bMax),
          normalColor);
  Mat abnormalColor = ~normalColor;
  Mat colorDefects;
  bitwise_and(abnormalColor, colorMask, colorDefects);

  // shiny edges of good nuts give thin lines of wrong color.
  // With 5x5 a good nut still had 3607 color pixels (good/157),
  // 9x9 removes the thin lines but keeps the real stains.
  Mat openKernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
  morphologyEx(colorDefects, colorDefects, MORPH_OPEN, openKernel);

  result.colorPixels = countNonZero(colorDefects);

  // limits come from the 242 good images: max perimeter 2261,
  // 0 gray pixels in the ring, at most 634 color pixels
  result.isDefective = result.perimeter > 2300 || result.ringPixels > 100 ||
                       result.colorPixels > 700;

  // red: gray pixels in the ring, blue: color stain, green: hole
  Mat resultImage = image.clone();
  resultImage.setTo(Scalar(0, 0, 255), grayPixels);
  resultImage.setTo(Scalar(255, 0, 0), colorDefects);
  circle(resultImage, nut.holeCenter, nut.holeRadius, Scalar(0, 255, 0), 2);

  Mat maskBgr;
  cvtColor(maskNoHole, maskBgr, COLOR_GRAY2BGR);

  hconcat(resultImage, maskBgr, comparison);

  result.processingOk = true;
  return result;
}
