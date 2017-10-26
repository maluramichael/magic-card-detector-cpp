#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
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

tesseract::TessBaseAPI* tess;

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

PointList findCardContour(const cv::Mat& inputImage) {// Find contours inside the edges image
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

PointList getPolygonFromHull(std::vector<cv::Point>& points, float x = 0.04) {
    // Get approximated polygon from hull
    auto epsilon = x * cv::arcLength(points, true);
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
    const int windowRowSize = 5;

    const std::string windows[] = {
            "dilate", "threshold", "working", "cardOutline", "homography",
            "title_threshold"
    };

    for (const auto& name:windows) {
        cv::namedWindow(name);
        cv::moveWindow(name, windowXPosition, windowYPosition);
        windowXPosition += windowSpacing;
        windowCount++;
        if (windowCount % windowRowSize == 0) {
            windowXPosition = 0;
            windowYPosition += windowSpacing;
        }
    }
}

template<typename I>
std::string n2hexstr(I w, size_t hex_len = sizeof(I) << 1) {
    static const char* digits = "0123456789ABCDEF";
    std::string rc(hex_len, '0');
    for (size_t i = 0, j = (hex_len - 1) * 4; i < hex_len; ++i, j -= 4)
        rc[i] = digits[(w >> j) & 0x0f];
    return rc;
}

cv::Mat getHash(cv::Mat& input) {
    cv::Mat image = input.clone();
//    cv::resize(image, image, cv::Size(9, 8)); // Reduce size
//    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY); // Convert to gray
//
//    long long hash = 0L;
//
//    int bit = 0;
//    for (int row = 0; row < image.rows; row++) {
//
//        for (int column = 0; column < image.cols - 1; column++) {
//            auto A = (int) image.data[row * 9 + column];
//            auto B = (int) image.data[row * 9 + column + 1];
//            if (A < B) {
//                hash |= 1 << bit;
//            }
//
//            bit++;
//        }
//
//    }
//
//    cv::resize(image, image, cv::Size(400, 400), 0, 0, cv::INTER_NEAREST);
//    return image;
//    cv::blur(image, image, cv::Size(3, 3));
//    cv::dilate(image, image, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

    auto algo = cv::img_hash::PHash::create();
    algo->compute(image, image);
    return image;
}

bool detectCard(cv::Mat& input, cv::Mat& output, Options& options) {
    cv::Mat imageWorking;
    cv::cvtColor(input, imageWorking, cv::COLOR_BGR2GRAY);
    cv::blur(imageWorking, imageWorking, cv::Size(3, 3));
    cv::dilate(imageWorking, imageWorking, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);
    if (options.showWindows) { cv::imshow("dilate", imageWorking); }

    cv::threshold(imageWorking, imageWorking, 100, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    if (options.showWindows) { cv::imshow("threshold", imageWorking); }

    cv::Canny(imageWorking, imageWorking, 50, 200);
    if (options.showWindows) { cv::imshow("working", imageWorking); }

    auto cardContourHull = findCardContour(imageWorking);

    // If contour contains any points
    if (!cardContourHull.empty()) {
        auto area = cv::contourArea(cardContourHull);
        if (area > 3000) {
            auto approximatedPolygon = getPolygonFromHull(cardContourHull);
            if (approximatedPolygon.size() == 4) {
                // Add better approximated polygon

                std::vector<PointList> listContainingCardContour;
                listContainingCardContour.push_back(approximatedPolygon);

                cv::Rect cardBoundingBox = cv::boundingRect(approximatedPolygon);
                cv::Mat subImage = cv::Mat(imageWorking, cardBoundingBox);
                auto biggestHull = findCardContour(subImage);
                
                subImage = cv::Mat(imageWorking, cardBoundingBox);
                std::vector<PointList> hulls;
                hulls.push_back(biggestHull);
                drawContours(subImage, hulls, 0, cv::Scalar(0, 255, 0), 3);

                cv::imshow("boundingBox", subImage);


                const int targetWidth = 600;
                const auto targetHeight = (int) (targetWidth * CARD_ASPECT);
                PointList targetRectangle;
                targetRectangle.emplace_back(targetWidth, 0); // Top Right
                targetRectangle.emplace_back(targetWidth, targetHeight); // Bottom Right
                targetRectangle.emplace_back(0, targetHeight); // Bottom Left
                targetRectangle.emplace_back(0, 0); // Top Left
                auto foundHomography = cv::findHomography(approximatedPolygon, targetRectangle);
                cv::warpPerspective(input, output, foundHomography, cv::Size(targetWidth, targetHeight));

                // Draw outline. Biggest contour
                cv::Mat imageCardOutline = input.clone();
                drawContours(imageCardOutline, listContainingCardContour, 0, cv::Scalar(0, 255, 0), 3);

                // auto rect = cv::minAreaRect(approximatedPolygon);
                // std::cout << "Bounding box angle " << rect.angle << '\n';
//                drawContours(imageCardOutline, listContainingCardContour, 0, cv::Scalar(0, 255, 0), 3);

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

bool detectTitle(cv::Mat& input, cv::Mat& output, Options& options) {
    const int titleHeight = 60;
    const int topPadding = 45;
    const int horizontalPadding = 50;
    const auto slice = cv::Rect(horizontalPadding,
                                topPadding,
                                input.size().width - horizontalPadding * 2,
                                titleHeight);

    cv::Mat imageWorking = cv::Mat(input, slice);
    cv::cvtColor(imageWorking, output, cv::COLOR_BGR2GRAY);
//    cv::blur(output, output, cv::Size(5, 5));
//    cv::dilate(output, output, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    cv::threshold(output, output, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    cv::blur(output, output, cv::Size(2, 2));
    if (options.showWindows) { cv::imshow("title_threshold", output); }

    cv::imwrite("title.png", output);

    return false;
}

bool getArtwork(cv::Mat& input, cv::Mat& output) {
    const float topPercentage = 0.114;
    const float leftPercentage = 0.084;
    const float heightPercentage = 0.445;

    const float left = input.size().width * leftPercentage;
    const float top = input.size().height * topPercentage;
    const float width = input.size().width - (left * 2);
    const float height = input.size().height * heightPercentage;

    output = cv::Mat(input, cv::Rect(
            (int) left,
            (int) top,
            (int) width,
            (int) height
    ));

    return true;
}

bool handleImage(cv::Mat& input, cv::Mat& output, Options& options) {
    std::cout << "Detect card > ";
    bool cardDetected = detectCard(input, output, options);

    if (cardDetected) {
        std::cout << "Card detected > ";
        cv::Mat imageArtwork;
        getArtwork(output, imageArtwork);
        cv::imshow("Artwork", imageArtwork);
        cv::Mat hash = getHash(imageArtwork);

        std::cout << "HASH > " << hash << "\n";
//        cv::Mat imageTitle;
//        bool titleDetected = detectTitle(output, imageTitle, options);
//
//        if (titleDetected) {
//            std::cout << "Title detected > ";
//            return true;
//        } else {
//            std::cout << "No title detected\n";
//            return false;
//        }

//        cv::Mat imageHashed = createDHash(output);
//        cv::imshow("HASH", imageHashed);
    } else {
        std::cout << "No card detected\n";
        return false;
    }

}


int main(int argc, const char* const* argv) {

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
    args::Flag argUseOcr(parser, "ocr", "Use tesseract", {"ocr"});

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

    generateWindows();

    Options options;
    options.imagePath = argInputPath.Get();
    options.outputPath = argOutputPath.Get();
    options.useWebcam = argUseWebcam.Get();
    options.saveDetectedCard = argSaveDetectedCard.Get();
    options.showWindows = argShowWindows.Get();
    options.cameraId = argCameraId.Get();
    options.inputDirectory = argInputDirectory.Get();

    options.useOCR = argUseOcr.Get();
    tess = new tesseract::TessBaseAPI();
    if (tess->Init(".", "mtg") != 0) {
        std::cout << "Could not initialize tesseract.\n";
        exit(1);
    }
    tess->SetPageSegMode(tesseract::PSM_SINGLE_LINE);

    std::map<std::string, cv::Mat> imagePaths;

    if (!options.inputDirectory.empty()) {
        boost::filesystem::path inputDirectoryPath(options.inputDirectory);

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
    }

    if (!options.imagePath.empty()) {
        imagePaths[options.imagePath] = cv::Mat();
    }

    if (!imagePaths.empty()) {
        bool running = true;

        auto currentImage = imagePaths.begin();
        cv::Mat detectedCard;

        if (options.showWindows) {
            while (running) {
                if (currentImage->second.rows == 0) {
                    std::cout << "Load " << currentImage->first << '\n';
                    cv::Mat inputFrame = cv::imread(currentImage->first);
                    cv::resize(inputFrame, inputFrame, cv::Size(0, 0), 0.2, 0.2);
                    imagePaths[currentImage->first] = inputFrame;
                }
                auto cardDetected = handleImage(imagePaths[currentImage->first], detectedCard, options);

                auto key = cv::waitKey();
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
                        currentImage++;
                        if (currentImage == imagePaths.end()) {
                            currentImage--;
                        }
                        break;
                    default:
                        if (key != -1) std::cout << key << " pressed\n";
                        break;
                }
            }
        } else {
            while (currentImage != imagePaths.end()) {
                std::cout << "Load " << currentImage->first << '\n';
                cv::Mat inputFrame = cv::imread(currentImage->first);
                // cv::resize(inputFrame, inputFrame, cv::Size(0, 0), 0.2, 0.2);
                handleImage(inputFrame, detectedCard, options);
                currentImage++;
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

    tess->End();

    return 0;
}