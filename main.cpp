#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <random>
#include <cmath>
#include "args.hxx"

int random(int a, int b) {
    thread_local std::mt19937 eng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(a, b);
    return dist(eng);
}

typedef std::vector<cv::Point> PointList;

const float CARD_WIDTH = 57;
const float CARD_HEIGHT = 82;
const float CARD_ASPECT = CARD_HEIGHT / CARD_WIDTH;

tesseract::TessBaseAPI *tess;

double angleBetween(cv::Point &A, cv::Point &B) {
    auto deltaX = B.x - A.x;
    auto deltaY = B.y - A.y;

    return atan2(deltaY, deltaX) * 180 / M_PI;
}

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
    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > 100) {
            cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
            cv::Rect appRect(boundingRect(cv::Mat(contours_poly[i])));
            if (appRect.width > appRect.height)
                boundRect.push_back(appRect);
        }
    }
    return boundRect;
}

cv::Mat handleImage(cv::Mat &input, bool showWindows) {
    cv::Mat frame = input.clone();
    cv::Mat output;
//    if (showWindows) { cv::imshow("original", frame); }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blur;
    cv::blur(gray, blur, cv::Size(3, 3));

    cv::Mat edges;
    cv::Canny(blur, edges, 100, 200);
//    if (showWindows) { cv::imshow("edges", edges); }

    // Find contours inside the edges image
    std::vector<PointList> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    std::vector<PointList> hulls(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> hull;
        cv::convexHull(cv::Mat(contours[i]), hull, false);
        hulls[i] = hull;
    }

    // Sort hulls by their area
    std::sort(hulls.begin(), hulls.end(), [](std::vector<cv::Point> first, std::vector<cv::Point> second) {
        auto areaA = cv::contourArea(first);
        auto areaB = cv::contourArea(second);
        return areaA > areaB;
    });

    // Remove all hulls except the biggest one
    while (hulls.size() > 1) {
        hulls.pop_back();
    }

    if (hulls.size() == 1) {
        auto area = cv::contourArea(hulls[0]);

        if (area > 3000) {

            // Get approximated polygon from hull
            auto epsilon = 0.1 * cv::arcLength(hulls[0], true);
            PointList approximatedPolygon;
            cv::approxPolyDP(hulls[0], approximatedPolygon, epsilon, true);

            // Remove current curvy hull
            hulls.pop_back();

            if (approximatedPolygon.size() == 4) {

                bool validRectangle = true;
//                for (int i = 0; i < 4; i += 2) {
//                    auto A = approximatedPolygon[i];
//                    auto B = approximatedPolygon[i + 1];
//
//                    auto angle = abs(angleBetween(A, B));
//                    std::cout << angle << '\n';
//                    if (angle <= 170) {
//                        validRectangle = false;
//                        break;
//                    }
//                }

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

                // Draw outline. Biggest contour
                cv::Scalar color = cv::Scalar(0, 255, 0);
                drawContours(frame, hulls, 0, color, 3, 8, std::vector<cv::Vec4i>(), 0, cv::Point());

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

                if (showWindows) { cv::imshow("homography", corrected); }


//                cv::imshow("title", title);
//                auto center = cv::Mat(corrected, cv::Rect(0, (int) (height * 0.5f), width, (int) (height * 0.15f)));
//                cv::imshow("center", center);
            }
        }
    }

    if (showWindows) { cv::imshow("contours", frame); }

    return output;
}

int main(int argc, const char *const *argv) {
    args::ArgumentParser parser("Card detector");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Group group(parser, "This group is all exclusive:", args::Group::Validators::Xor);
    args::ValueFlag<std::string> imagePath(group, "image", "Image input path", {'i'});
    args::Flag showWindows(parser, "show", "Show windows", {'s'});
    args::Flag useWebcam(group, "webcam", "Use webcam", {'w'});
    args::ValueFlag<int> cameraId(parser, "camera", "Camera id", {'c'});
    args::Flag useOCR(parser, "ocr", "Use tesseract", {'o'});

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
    if (useOCR) {
        tess = new tesseract::TessBaseAPI();
        if (tess->Init(nullptr, "deu") != 0) {
            std::cout << "Could not initialize tesseract.\n";
            exit(1);
        }
        tess->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    }

    if (showWindows || useWebcam) {
//        cv::namedWindow("original", cv::WINDOW_NORMAL);
//        cv::namedWindow("edges", cv::WINDOW_NORMAL);
        cv::namedWindow("contours", cv::WINDOW_NORMAL);
//        cv::namedWindow("title", cv::WINDOW_NORMAL);

//        cv::namedWindow("center", cv::WINDOW_NORMAL);

//        cv::moveWindow("original", 0, 0);
//        cv::moveWindow("edges", 640, 0);
        cv::moveWindow("contours", 640, 450);
    }

    if (imagePath) {
        std::string &inputFilename = imagePath.Get();
        std::string outputFilename = inputFilename;
        outputFilename = outputFilename.substr(0, outputFilename.rfind('.'));
        outputFilename += "_converted.png";

        std::cout << "Load image " << args::get(imagePath) << '\n';
        cv::Mat frame = cv::imread(args::get(imagePath));
        cv::resize(frame, frame, cv::Size(0, 0), 0.2, 0.2);

        if (showWindows) {
            handleImage(frame, showWindows);
            cv::waitKey();
        } else {
            cv::Mat outputFrame = handleImage(frame, showWindows);
            std::cout << "Save output image " << outputFilename << '\n';
            cv::imwrite(outputFilename, outputFrame);
        }
    } else if (useWebcam) {
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
            handleImage(frame, showWindows || useWebcam);

            if (cv::waitKey(1) == 27) { break; }
        }
    }

    if (useOCR) {
        tess->End();
    }

    return 0;
}