#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

double euclideanDist(cv::Point &p, cv::Point &q) {
    cv::Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

void handleImage(cv::Mat &frame) {
    cv::resize(frame, frame, cv::Size(0, 0), 0.2, 0.2);

    cv::Mat output;
    cv::imshow("original", frame);

    cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
    cv::imshow("gray", output);

    cv::blur(output, output, cv::Size(4, 4));
    cv::imshow("blured", output);

    cv::Canny(output, output, 100, 200);
    cv::imshow("edges", output);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // Find convex hull object for each container
    std::vector<std::vector<cv::Point>> hulls(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> hull;
        cv::convexHull(cv::Mat(contours[i]), hull, false);

        hulls[i] = hull;
    }

    /// Draw contours
    cv::Mat drawing = cv::Mat::zeros(output.size(), CV_8UC3);
    for (int i = 0; i < hulls.size(); i++) {
        cv::Scalar color = cv::Scalar(255, 255, 255);
//        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
        drawContours(drawing, hulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
    }

    std::vector<cv::Vec4i> lines;
    double rho = 2;
    double theta = CV_PI / 180;
    int threshold = 100;
    double minLineLength = 3;
    double maxLineGap = 30;

    cv::HoughLinesP(output, lines, rho, theta, threshold, minLineLength, maxLineGap);

    std::sort(lines.begin(), lines.end(), [](cv::Vec4i lineA, cv::Vec4i lineB) {
        cv::Point lineAPointA(lineA[0], lineA[1]);
        cv::Point lineAPointB(lineA[2], lineA[3]);
        auto distanceA = euclideanDist(lineAPointA, lineAPointB);

        cv::Point lineBPointA(lineB[0], lineB[1]);
        cv::Point lineBPointB(lineB[2], lineB[3]);
        auto distanceB = euclideanDist(lineBPointA, lineBPointB);

        return distanceA > distanceB;
    });

    for (size_t i = 0; i < lines.size() && i < 10; i++) {
        cv::Vec4i l = lines[i];
        cv::Point a(l[0], l[1]);
        cv::Point b(l[2], l[3]);

        cv::line(drawing, a, b, cv::Scalar(0, 0, 255), 3, 2);
    }


    cv::imshow("contours", drawing);
}


int main(int argc, char **argv) {
    bool useImage = argc == 2;

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

    if (useImage) {
        std::cout << "Load image\n";
        cv::Mat frame = cv::imread(argv[1]);
        handleImage(frame);
        cv::waitKey();
    } else {
        std::cout << "Capture video\n";
        cv::VideoCapture capture(1);

        if (!capture.isOpened()) {
            return 1;
        }

        for (;;) {
            cv::Mat frame;
            capture >> frame;

            if (frame.empty()) break;

            handleImage(frame);

            if (cv::waitKey(1) == 27) break;
        }
    }


    return 0;
}