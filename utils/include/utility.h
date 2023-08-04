#ifndef UTILITY_H
#define UTILITY_H
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <iomanip> //setw

using namespace std;

//https://stackoverflow.com/questions/249701/why-arent-my-compile-guards-preventing-multiple-definition-inclusions
//a header file should not contain definition of variables or functions

std::string type2str(int type);
void stringReplace(string& base, string from, string to);
void pad(cv::Mat& img, int padt, int padb, int padl, int padr, int mode = cv::BORDER_CONSTANT);
void ConcatHorizontal(vector<cv::Mat>& imgs, cv::Mat& target);
void Show(cv::Mat& img, string title = "Display Window", bool save = false);
void ShowHorizontal(vector<cv::Mat>& imgs, string title = "Display Window", bool save = false);
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6);
//from CH4
void PrintVector(vector<double>& row);
//from CH4
void PrintMatrix(vector<vector<double>>& matrix);
//from CH4
void SwapRow(vector<double>& row1, vector<double>& row2);
//from CH4
void MultiplyRow(vector<double>& row, double ratio);
//from CH4
void MultiplySubstractRow(vector<double>& row1, vector<double>& row2, double ratio);
//from CH4
bool InvMat(vector<vector<double>>& matrix);
//from CH4
bool ProdMat(vector<vector<double>>& mat1, vector<vector<double>>& mat2, vector<vector<double>>& res);

#include "to_string_with_precision.tpp"
#endif
