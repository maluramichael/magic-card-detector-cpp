#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <random>
#include <cmath>
#include <boost/filesystem.hpp>

#include "args.hxx"

typedef std::vector<cv::Point> PointList;
const float CARD_WIDTH = 57;
const float CARD_HEIGHT = 82;
const float CARD_ASPECT = CARD_HEIGHT / CARD_WIDTH;

#ifdef USE_OCR
tesseract::TessBaseAPI* tess;
#endif

std::vector<cv::Rect> detectLetters(cv::Mat& img) {
    std::vector<cv::Rect> boundRect;
    cv::Mat grayImage, sobelImage, thresholdImage, element;
    cvtColor(img, grayImage, CV_BGR2GRAY);
    cv::Sobel(grayImage, sobelImage, CV_8U, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::threshold(sobelImage, thresholdImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    element = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    cv::morphologyEx(thresholdImage, thresholdImage, CV_MOP_CLOSE, element); //Does the trick
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(thresholdImage, contours, 0, 1);
    std::vector<std::vector<cv::Point> > contours_poly(contours.size());
    // TODO: replace with foreach
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() > 100) {
            cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
            cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));
            if (appRect.width > appRect.height) {
                boundRect.push_back(appRect);
            }
        }
    }
    return boundRect;
}

std::vector<PointList> findCardContour(const cv::Mat& inputImage) {// Find contours inside the edges image
    std::vector<PointList> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(inputImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    std::vector<PointList> hulls(contours.size());
    // TODO: replace with foreach
    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> hull;
        convexHull(cv::Mat(contours[i]), hull, false);
        hulls[i] = hull;
    }
    
    // Sort hulls by their area
    sort(hulls.begin(), hulls.end(), [](std::vector<cv::Point> first, std::vector<cv::Point> second) {
      auto areaA = contourArea(first);
      auto areaB = contourArea(second);
      return areaA > areaB;
    });
    
    // Remove all hulls except the biggest one
    while (hulls.size() > 1) {
        hulls.pop_back();
    }
    return hulls;
}

PointList getPolygonFromHull(std::vector<cv::Point>& points) {
    // Get approximated polygon from hull
    auto epsilon = 0.04 * cv::arcLength(points, true);
    PointList approximatedPolygon;
    cv::approxPolyDP(points, approximatedPolygon, epsilon, true);
    return approximatedPolygon;
}

struct Options {
  std::string imagePath = "";
  std::string outputPath = "";
  bool useWebcam = false;
  bool showWindows = false;
  bool useOCR = false;
  bool saveDetectedCard = false;
};

cv::Mat detectCard(cv::Mat& input, Options& options) {
    cv::Mat frame = input.clone();
    cv::Mat output;
    if (options.showWindows) { cv::imshow("original", frame); }
    
    cv::Mat working;
    cv::cvtColor(frame, working, cv::COLOR_BGR2GRAY);
    
    cv::blur(working, working, cv::Size(6, 6));
//    cv::imwrite("debug/blured.png", working);
    if (options.showWindows) { cv::imshow("blured", working); }
    
    cv::dilate(working, working, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);
    if (options.showWindows) { cv::imshow("dilate", working); }
    
    cv::threshold(working, working, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    //cv::adaptiveThreshold(working, working, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 105, 1);
//    cv::imwrite("debug/threshold.png", working);
    if (options.showWindows) { cv::imshow("threshold", working); }
    
    cv::Canny(working, working, 100, 200);
//    cv::imwrite("debug/canny.png", working);
    if (options.showWindows) { cv::imshow("working", working); }
    
    output = working.clone();
    
    auto hulls = findCardContour(output);
    
    if (hulls.size() == 1) {
        auto area = cv::contourArea(hulls[0]);
        
        if (area > 3000) {
            
            auto approximatedPolygon = getPolygonFromHull(hulls[0]);
            
            // Remove current curvy hull
            hulls.pop_back();
            
            if (approximatedPolygon.size() == 4) {
                // Add better approximated polygon
                hulls.push_back(approximatedPolygon);
                
                const int width = 600;
                const auto height = (int) (width * CARD_ASPECT);
                PointList destinationPoints;
                destinationPoints.emplace_back(width, 0); // Top Right
                destinationPoints.emplace_back(width, height); // Bottom Right
                destinationPoints.emplace_back(0, height); // Bottom Left
                destinationPoints.emplace_back(0, 0); // Top Left
                auto h = cv::findHomography(approximatedPolygon, destinationPoints);
                cv::Mat corrected;
                cv::warpPerspective(frame, corrected, h, cv::Size(width, height));
//                cv::imwrite("debug/corrected.png", corrected);
                
                // Draw outline. Biggest contour
                cv::Mat contours = frame.clone();
                cv::Scalar color = cv::Scalar(0, 255, 0);
                drawContours(contours, hulls, 0, color, 3, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
//                cv::imwrite("debug/contours.png", contours);
                
                auto title = cv::Mat(corrected, cv::Rect(0, 0, width, (int) (height * 0.15f)));
//                auto textBoxes = detectLetters(title);
//                for (const auto &box: textBoxes) {
//                    cv::rectangle(title, box, cv::Scalar(255, 255, 0), 3, 8, 0);
//                    if (tess != nullptr) {
//                        auto subBox = cv::Mat(title, box);
//                        tess->SetImage((uchar *) subBox.data, subBox.size().width, subBox.size().height,
//                                       subBox.channels(), subBox.step1());
//                        tess->Recognize(0);
//                        std::cout << "Title: " << tess->GetUTF8Text() << '\n';
//                    }
//                }
                
                if (options.showWindows) {
                    cv::imshow("contours", contours);
                    cv::imshow("homography", corrected);
                }
                return corrected;
                //cv::imshow("title", title);
//                auto center = cv::Mat(corrected, cv::Rect(0, (int) (height * 0.5f), width, (int) (height * 0.15f)));
//                cv::imshow("center", center);
            }
        }
    }
    
    return cv::Mat();
}

int main(int argc, const char* const* argv) {
    args::ArgumentParser parser("Card detector");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Group group(parser, "This group is all exclusive:", args::Group::Validators::Xor);
    args::ValueFlag<std::string> argImagePath(group, "image", "Image input path", {"image"});
    args::ValueFlag<std::string> argOutputPath(parser, "output path", "Path where generated images are saved", {"output"});
    args::Flag argShowWindows(parser, "show", "Show windows", {"windows"});
    args::Flag argSaveDetectedCard(parser, "write", "Save detected card", {"save"});
    args::Flag argUseWebcam(group, "webcam", "Use webcam", {"webcam"});
    args::ValueFlag<int> argCameraId(parser, "camera", "Camera id", {"cameraid"});

#ifdef USE_OCR
    args::Flag argUseOcr(parser, "ocr", "Use tesseract", {"ocr"});
#endif
    
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help&) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    } catch (args::ValidationError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    
    Options options;
    options.imagePath = argImagePath.Get();
    options.outputPath = argOutputPath.Get();
    options.useWebcam = argUseWebcam.Get();
    options.saveDetectedCard = argSaveDetectedCard.Get();
    options.showWindows = argShowWindows.Get();

#ifdef USE_OCR
    options.useOCR = argUseOcr.Get();
    if (options.useOCR) {
        tess = new tesseract::TessBaseAPI();
        if (tess->Init(nullptr, "deu") != 0) {
            std::cout << "Could not initialize tesseract.\n";
            exit(1);
        }
        tess->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    }
#endif
    
    if (!options.imagePath.empty()) {
        std::string inputFilename = options.imagePath;
        std::string outputFilename = inputFilename;
        boost::filesystem::path currentPath(boost::filesystem::current_path());
        outputFilename = outputFilename.substr(0, outputFilename.rfind('.'));
        outputFilename = outputFilename.substr(outputFilename.rfind('/') + 1);
        outputFilename += "_detected.png";
        
        if (!options.outputPath.empty()) {
            currentPath = currentPath / options.outputPath;
        }
        currentPath = currentPath / outputFilename;
        
        cv::Mat inputFrame = cv::imread(options.imagePath);
        cv::resize(inputFrame, inputFrame, cv::Size(0, 0), 0.2, 0.2);
        
        if (options.showWindows) {
            cv::Mat detectedCard = detectCard(inputFrame, options);
            if (detectedCard.rows == 0 || detectedCard.cols == 0) {
                std::cout << "No card detected\n";
            }
            cv::waitKey();
        } else {
            cv::Mat detectedCard = detectCard(inputFrame, options);
            if (detectedCard.rows == 0 || detectedCard.cols == 0) {
                std::cout << "No card detected\n";
            } else {
                if (options.saveDetectedCard) {
                    std::cout << "CARD DETECTED! " << currentPath << "\n";
                    cv::imwrite(currentPath.string(), detectedCard);
                } else {
                    std::cout << "CARD DETECTED!\n";
                }
            }
        }
    } else if (options.useWebcam) {
        std::cout << "Capture video\n";
        cv::VideoCapture capture(1);
        
        if (!capture.isOpened()) {
            return 1;
        }
        
        for (;;) {
            cv::Mat frame;
            capture >> frame;
            
            if (frame.empty()) { break; }
            cv::resize(frame, frame, cv::Size(0, 0), 0.4, 0.4);
            detectCard(frame, options);
            
            if (cv::waitKey(1) == 27) { break; }
        }
    }

#ifdef USE_OCR
    if (argUseOcr) {
        tess->End();
    }
#endif
    
    return 0;
}