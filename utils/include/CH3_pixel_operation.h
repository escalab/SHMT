#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include "utility.h"

using namespace std;

//cannot set default value for function declaration?
bool GenHist(cv::Mat& img, vector<double>& pdHist, int n = 256, bool normalize = true);
bool LinTran(cv::Mat& img, double dFa, double dFb);
bool LogTran(cv::Mat& img, double dC = 10.0);
bool GammaTran(cv::Mat& img, double gamma, double comp);
bool Threshold(cv::Mat& img, int nThres);
bool ParLinTran(cv::Mat& img, int x1, int x2, int y1, int y2);
bool GetHisteqMap(cv::Mat& img, vector<int>& histeqMap);
bool Histeq(cv::Mat& img);
void GetInverseEqHist(vector<double>& hist, vector<int>& inverseHistEqMap);
bool Histst(cv::Mat& img, vector<double> stdHist);
bool Histst(cv::Mat& img, cv::Mat& stdImg);
void DrawHist(vector<double>& hist, cv::Mat& histImage, int img_h = 400);
