#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <random>
#include <cmath>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <string>
#include "args.hxx"

typedef std::vector<cv::Point> PointList;
const float CARD_WIDTH = 57;
const float CARD_HEIGHT = 82;
const float CARD_ASPECT = CARD_HEIGHT / CARD_WIDTH;

#ifdef USE_OCR
tesseract::TessBaseAPI* tess;
#endif

std::vector<cv::Rect> detectLetters(cv::Mat &img) {
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

PointList findCardContour(const cv::Mat &inputImage) {// Find contours inside the edges image
    std::vector<PointList> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(inputImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
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

    if (hulls.size() == 0) { return PointList(); }
    return hulls[0];
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
    int cameraId = 0;
    std::string inputDirectory = "";
};

void generateWindows() {

    int windowXPosition = 0;
    int windowYPosition = 0;
    int windowCount = 0;
    const int windowSpacing = 500;
    const int windowRowSize = 3;

    const std::string windows[] = {
            "original", "blured", "dilate", "threshold", "working", "cardOutline", "homography"
    };

    for (const auto name:windows) {
        cv::namedWindow(name);
        cv::moveWindow(name, windowXPosition, windowYPosition);
        windowXPosition += windowSpacing;
        if (windowCount % windowRowSize == 0 && windowCount != 0) {
            windowXPosition = 0;
            windowYPosition += windowSpacing;
        }
        windowCount++;
    }
}

bool detectCard(cv::Mat& input, cv::Mat& output, Options& options) {

    cv::Mat imageOriginal = input.clone();
    cv::Mat imageOutput;
    if (options.showWindows) { cv::imshow("original", imageOriginal); }

    cv::Mat imageWokring;
    cv::cvtColor(imageOriginal, imageWokring, cv::COLOR_BGR2GRAY);

    cv::blur(imageWokring, imageWokring, cv::Size(3,3));
    if (options.showWindows) { cv::imshow("blured", imageWokring); }

    cv::dilate(imageWokring, imageWokring, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);
    if (options.showWindows) { cv::imshow("dilate", imageWokring); }

    cv::threshold(imageWokring, imageWokring, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    if (options.showWindows) { cv::imshow("threshold", imageWokring); }

    cv::Canny(imageWokring, imageWokring, 50, 200);
    if (options.showWindows) { cv::imshow("working", imageWokring); }

    auto cardContourHull = findCardContour(imageWokring);

    // If contour contains any points
    if (cardContourHull.size() != 0) {
        auto area = cv::contourArea(cardContourHull);
        if (area > 3000) {
            auto approximatedPolygon = getPolygonFromHull(cardContourHull);
            if (approximatedPolygon.size() == 4) {
                // Add better approximated polygon

                std::vector<PointList> listContainingCardContour;
                listContainingCardContour.push_back(approximatedPolygon);

                const int targetWidth = 600;
                const auto targetHeight = (int) (targetWidth * CARD_ASPECT);
                PointList targetRectangle;
                targetRectangle.emplace_back(targetWidth, 0); // Top Right
                targetRectangle.emplace_back(targetWidth, targetHeight); // Bottom Right
                targetRectangle.emplace_back(0, targetHeight); // Bottom Left
                targetRectangle.emplace_back(0, 0); // Top Left
                auto foundHomography = cv::findHomography(approximatedPolygon, targetRectangle);
                cv::warpPerspective(imageOriginal, output, foundHomography, cv::Size(targetWidth, targetHeight));

                // Draw outline. Biggest contour
                cv::Mat imageCardOutline = imageOriginal.clone();
                drawContours(imageCardOutline, listContainingCardContour, 0, cv::Scalar(0, 255, 0), 3);

                // auto rect = cv::minAreaRect(approximatedPolygon);
                // std::cout << "Bounding box angle " << rect.angle << '\n';
//                drawContours(imageCardOutline, listContainingCardContour, 0, cv::Scalar(0, 255, 0), 3);

                // auto title = cv::Mat(imageCorrectedPerspective, cv::Rect(0, 0, targetWidth, (int) (targetHeight * 0.15f)));
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
                    cv::imshow("cardOutline", imageCardOutline);
                    cv::imshow("homography", output);
                }

                return true;
            }
        }
    }
    return false;
}

bool detectTitle(cv::Mat& input, cv::Mat& output, Options& options){
    
    cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);

    cv::blur(output, output, cv::Size(3,3));
    if (options.showWindows) { cv::imshow("description_blured", output); }

    cv::dilate(output, output, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);
    if (options.showWindows) { cv::imshow("description_dilate", output); }

    cv::threshold(output, output, 50, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    if (options.showWindows) { cv::imshow("description_threshold", output); }

    cv::Canny(output, output, 50, 200);
    if (options.showWindows) { cv::imshow("description_working", output); }

    std::vector<PointList> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    std::vector<PointList> polygons;

    for (const auto& contour: contours) {
        std::vector<cv::Point> points;
        cv::convexHull(cv::Mat(contour), points, false);
        auto area = contourArea(points);
        if (area > 3000){
            auto approximatedPolygon = getPolygonFromHull(points);
            if (approximatedPolygon.size() == 4) {
                polygons.push_back(approximatedPolygon);
            }
        }
    }

    std::cout << "Found " << polygons.size() << " polygons\n";    

    // Sort polygons by their area
    sort(polygons.begin(), polygons.end(), [](std::vector<cv::Point> first, std::vector<cv::Point> second) {
        auto areaA = contourArea(first);
        auto areaB = contourArea(second);
        return areaA > areaB;
    });

    // Draw outline. Biggest contour
    cv::Mat imageOutline = input.clone();
    for(int index = 0; index < polygons.size(); index++) {
        drawContours(imageOutline, polygons, index, cv::Scalar(0, 255, 0), 3);
    }

    if (options.showWindows) { cv::imshow("description_working", imageOutline); }

    return false;
}

cv::Mat handleImage(cv::Mat &input, Options &options) {
    cv::Mat imageDetectedCard;
    bool cardDetected = detectCard(input, imageDetectedCard, options);

    if (cardDetected) {
        cv::Mat imageTitle;
        detectTitle(imageDetectedCard, imageTitle, options);
    }

    return cv::Mat();
}


int main(int argc, const char *const *argv) {

    args::ArgumentParser parser("Card detector");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Group group(parser, "This group is all exclusive:", args::Group::Validators::Xor);
    args::ValueFlag<std::string> argInputPath(group, "image", "Image input path", {"image"});
    args::PositionalList<std::string> argInputList(group, "images", "List of images");
    args::ValueFlag<std::string> argInputDirectory(group, "inputdirectory", "Directory containing images",
                                                   {"inputdirectory"});
    args::ValueFlag<std::string> argOutputPath(parser, "output path", "Path where generated images are saved",
                                               {"output"});
    args::Flag argShowWindows(parser, "show", "Show windows", {"windows"});
    args::Flag argSaveDetectedCard(parser, "write", "Save detected card", {"save"});
    args::Flag argUseWebcam(group, "webcam", "Use webcam", {"webcam"});
    args::ValueFlag<int> argCameraId(parser, "camera", "Camera id", {"cameraid"});

#ifdef USE_OCR
    args::Flag argUseOcr(parser, "ocr", "Use tesseract", {"ocr"});
#endif

    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help &) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    } catch (args::ValidationError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    generateWindows();

    Options options;
    options.imagePath = argInputPath.Get();
    options.outputPath = argOutputPath.Get();
    options.useWebcam = argUseWebcam.Get();
    options.saveDetectedCard = argSaveDetectedCard.Get();
    options.showWindows = argShowWindows.Get();
    options.cameraId = argCameraId.Get();
    options.inputDirectory = argInputDirectory.Get();

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

    if (!options.inputDirectory.empty()) {
        boost::filesystem::path inputDirectoryPath(options.inputDirectory);

        std::map<std::string, cv::Mat> imagePaths;

        // cycle through the directory
        for (boost::filesystem::directory_iterator iterator(inputDirectoryPath);
             iterator != boost::filesystem::directory_iterator();
             ++iterator) {
            if (boost::filesystem::is_regular_file(iterator->path())) {
                std::string currentFile = iterator->path().string();
                std::string extension = boost::filesystem::extension(currentFile);
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

                if (extension == ".jpg" || extension == ".png") {
                    std::cout << "Found " << currentFile << std::endl;
//                    imagePaths.push_back(currentFile);
                    imagePaths[currentFile] = cv::Mat();
                }
            }
        }


        if (imagePaths.size() > 0) {
            bool running = true;

            std::map<std::string, cv::Mat>::iterator currentImage = imagePaths.begin();

            while (running) {

                if (currentImage->second.rows == 0) {
                    std::cout << "Load " << currentImage->first << '\n';
                    cv::Mat inputFrame = cv::imread(currentImage->first);
                    cv::resize(inputFrame, inputFrame, cv::Size(0, 0), 0.2, 0.2);
                    imagePaths[currentImage->first] = inputFrame;
                }
                handleImage(imagePaths[currentImage->first], options);

                auto key = cv::waitKey(1000);
                switch (key) {
                    case 27: // esc
                        running = false;
                        break;
                    case 2: // left
                        if (currentImage != imagePaths.begin()) {
                            currentImage--;
                        }
                        break;
                    case 3: // right
                        if (currentImage != imagePaths.end()) {
                            currentImage++;
                        }
                        break;
                    default:
                        if (key != -1) std::cout << key << " pressed\n";
                        break;
                }
            }
        }
    }
//        else {
//            std::string inputFilename = options.imagePath;
//            boost::filesystem::path currentPath(boost::filesystem::current_path());
//
//            cv::Mat inputFrame = cv::imread(options.imagePath);
//            cv::resize(inputFrame, inputFrame, cv::Size(0, 0), 0.2, 0.2);
//
//            cv::Mat detectedCard = handleImage(inputFrame, options);
//            if (detectedCard.rows == 0 || detectedCard.cols == 0) {
//                std::cout << "No card detected\n";
//            } else {
//                if (options.saveDetectedCard) {
//                    std::cout << "CARD DETECTED! " << currentPath << "\n";
//                    cv::imwrite(currentPath.string(), detectedCard);
//                } else {
//                    std::cout << "CARD DETECTED!\n";
//                }
//            }
//        }
//    } else if (options.useWebcam) {
//        std::cout << "Capture video\n";
//        cv::VideoCapture capture(options.cameraId);
//
//        if (!capture.isOpened()) {
//            return 1;
//        }
//
//        for (;;) {
//            cv::Mat frame;
//            capture >> frame;
//
//            if (frame.empty()) { break; }
//            cv::resize(frame, frame, cv::Size(0, 0), 0.4, 0.4);
//            handleImage(frame, options);
//
//            if (cv::waitKey(1) == 27) { break; }
//        }
//    }

#ifdef USE_OCR
    if (argUseOcr) {
        tess->End();
    }
#endif

    return 0;
}