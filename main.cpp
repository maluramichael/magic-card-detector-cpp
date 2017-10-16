#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "args.hxx"

double euclideanDist(cv::Point& p, cv::Point& q) {
  cv::Point diff = p - q;
  return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

cv::Mat handleImage(cv::Mat& frame, bool showWindows) {
  cv::resize(frame, frame, cv::Size(0, 0), 0.2, 0.2);
  
  cv::Mat output;
  if (showWindows) { cv::imshow("original", frame); }
  
  cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
  if (showWindows) { cv::imshow("gray", output); }
  
  cv::blur(output, output, cv::Size(2, 2));
  if (showWindows) { cv::imshow("blured", output); }
  
  cv::Canny(output, output, 100, 200);
  if (showWindows) { cv::imshow("edges", output); }
  
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  std::vector<std::vector<cv::Point>> hulls(contours.size());
  for (int i = 0; i < contours.size(); i++) {
    std::vector<cv::Point> hull;
    cv::convexHull(cv::Mat(contours[i]), hull, false);
    
    hulls[i] = hull;
  }
  
  for (int i = 0; i < hulls.size(); i++) {
    cv::Scalar color = cv::Scalar(255, 255, 255);
    drawContours(output, hulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
  }
  
  //std::vector<cv::Vec4i> lines;
  //double rho = 2;
  //double theta = CV_PI / 180;
  //int threshold = 100;
  //double minLineLength = 3;
  //double maxLineGap = 30;
  //
  //cv::HoughLinesP(output, lines, rho, theta, threshold, minLineLength, maxLineGap);
  //
  //std::sort(lines.begin(), lines.end(), [](cv::Vec4i lineA, cv::Vec4i lineB) {
  //    cv::Point lineAPointA(lineA[0], lineA[1]);
  //    cv::Point lineAPointB(lineA[2], lineA[3]);
  //    auto distanceA = euclideanDist(lineAPointA, lineAPointB);
  //
  //    cv::Point lineBPointA(lineB[0], lineB[1]);
  //    cv::Point lineBPointB(lineB[2], lineB[3]);
  //    auto distanceB = euclideanDist(lineBPointA, lineBPointB);
  //
  //    return distanceA > distanceB;
  //});
  //
  //for (size_t i = 0; i < lines.size() && i < 10; i++) {
  //    cv::Vec4i l = lines[i];
  //    cv::Point a(l[0], l[1]);
  //    cv::Point b(l[2], l[3]);
  //
  //    cv::line(drawing, a, b, cv::Scalar(0, 0, 255), 3, 2);
  //}
  
  if (showWindows) { cv::imshow("contours", output); }
  
  return output;
}

int main(int argc, const char* const* argv) {
  args::ArgumentParser parser("Card detector");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::Group group(parser, "This group is all exclusive:", args::Group::Validators::Xor);
  args::ValueFlag<std::string> imagePath(group, "image", "Image input path", {'i'});
  args::Flag showWindows(parser, "show", "Show windows", {'s'});
  args::Flag useWebcam(group, "webcam", "Use webcam", {'w'});
  args::ValueFlag<int> cameraId(parser, "camera", "Camera id", {'c'});
  
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  } catch (args::ValidationError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  
  if (showWindows || useWebcam) {
    cv::namedWindow("original", cv::WINDOW_NORMAL);
    cv::namedWindow("gray", cv::WINDOW_NORMAL);
    cv::namedWindow("blured", cv::WINDOW_NORMAL);
    cv::namedWindow("edges", cv::WINDOW_NORMAL);
    cv::namedWindow("contours", cv::WINDOW_NORMAL);
    
    cv::moveWindow("original", 0, 0);
    cv::moveWindow("gray", 0, 450);
    cv::moveWindow("blured", 0, 880);
    cv::moveWindow("edges", 640, 0);
    cv::moveWindow("contours", 640, 450);
  }
  
  if (imagePath) {
    std::string& inputFilename = imagePath.Get();
    std::string outputFilename = inputFilename;
    outputFilename = outputFilename.substr(0, outputFilename.rfind('.'));
    outputFilename += "_converted.png";
    
    std::cout << "Load image " << args::get(imagePath) << '\n';
    cv::Mat frame = cv::imread(args::get(imagePath));
    cv::Mat outputFrame = handleImage(frame, showWindows);
    
    if (!showWindows) {
      std::cout << "Save output image " << outputFilename << '\n';
      cv::imwrite(outputFilename, outputFrame);
    }
  } else if (useWebcam) {
    std::cout << "Capture video\n";
    cv::VideoCapture capture(0);
    
    if (!capture.isOpened()) {
      return 1;
    }
    
    for (;;) {
      cv::Mat frame;
      capture >> frame;
      
      if (frame.empty()) { break; }
      
      handleImage(frame, showWindows || useWebcam);
      
      if (cv::waitKey(1) == 27) { break; }
    }
  }
  
  return 0;
}