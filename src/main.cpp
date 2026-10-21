#include "defect_detector.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

/*
 * runFolder
 * - checks every .png in the folder
 * - writes one line per image to results.csv
 * - prints how many were DEFECT / OK at the end
 * No windows here, otherwise 220 windows would open.
 */
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

    // continue, not return: one broken image shouldn't stop the whole folder
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
  // usage: ./mtest <image>  -> prints the result and shows the image
  //        ./mtest <folder> -> checks all images, writes results.csv
  if (argc < 2) {
    cerr << "Usage: ./mtest <image_or_folder>\n";
    return -1;
  }

  string path = argv[1];

  if (filesystem::is_directory(path)) {
    return runFolder(path);
  }

  Mat image = imread(path);

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

  // saved next to the program (build/), handy for the README images
  if (!imwrite("nut_result.png", comparison)) {
    cerr << "Could not save the result.\n";
    return -1;
  }

  imshow("Result", comparison);
  waitKey(0);
  return 0;
}
