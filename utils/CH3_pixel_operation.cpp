#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include "utility.h"
#include "CH3_pixel_operation.h"

using namespace std;
using namespace cv;

bool GenHist(cv::Mat& img, vector<double>& pdHist, int n, bool normalize){
    //p.68
    //output is probability range from 0 to 1
    if(type2str(img.type()) != "8UC1") return false;
    if(n <= 0 || n > 256) return false;
    pdHist = vector<double>(n, 0);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            int val = (int)img.at<uchar>(i, j);
            pdHist[(int)val * (n/256.0)]++;
        }
    }

    int pixelCount = img.rows * img.cols;

    if(normalize){
        for(int i = 0; i < pdHist.size(); i++){
            pdHist[i] /= (double)pixelCount;
        }
    }
    return true;
};

bool LinTran(cv::Mat& img, double dFa, double dFb){
    //p.73
    if(type2str(img.type()) != "8UC1") return false;
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            double val = img.at<uchar>(i, j);
            val =  (val * dFa + dFb);
            img.at<uchar>(i, j) = min(max((int)val, 0), 255); 
        }
    }
    return true;
};

bool LogTran(cv::Mat& img, double dC){
    //p.75
    if(type2str(img.type()) != "8UC1") return false;
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            double val = img.at<uchar>(i, j);
            val = dC * log(val+1);
            img.at<uchar>(i, j) = min(max((int)val, 0), 255); 
        }
    }
    return true;
};

bool GammaTran(cv::Mat& img, double gamma, double comp){
    //p.79
    //gamma can be 0.75, 1, 1.5...
    if(type2str(img.type()) != "8UC1") return false;
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            double val = img.at<uchar>(i, j);
            //compensate
            val += comp;
            //normalize
            val /= 255.0;
            val = pow(val, gamma);
            //denormalize
            val *= 255.0;
            img.at<uchar>(i, j) = val;
        }
    }
    return true;
};

bool Threshold(cv::Mat& img, int nThres){
    //p.82
    if(type2str(img.type()) != "8UC1") return false;
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            img.at<uchar>(i, j) = (img.at<uchar>(i, j) < nThres) ? 0 : 255;
            //cout << (int)img.at<uchar>(i, j) << " ";
        }
        //cout << endl;
    }
    return true;
};

bool ParLinTran(cv::Mat& img, int x1, int x2, int y1, int y2){
    //p.89
    if(type2str(img.type()) != "8UC1") return false;
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            double val = img.at<uchar>(i, j);
            double slope;
            if(val < x1){
                slope = (double)y1/x1;
                val *= slope;
            }else if(val < x2){
                //x1 <= val < x2
                slope = (double)(y2-y1)/(x2-x1);
                val = (val-x1) * slope + y1;
            }else{
                //val >= x2
                slope = (double)(255-y2)/(255-x2);
                val = (val-x2) * slope + y2;
            }
            img.at<uchar>(i, j) = min(max((int)val, 0), 255); 
            img.at<uchar>(i, j) = val;
        }
    }
    return true;
};

/* added function */
double accumulate(vector<double>::iterator begin, vector<double>::iterator end, double sum){
    for (std::vector<double>::iterator it = begin ; it != end; ++it){
        sum += *it;
    }
    return sum;
}

bool GetHisteqMap(cv::Mat& img, vector<int>& histeqMap){
    //utility function
    if(type2str(img.type()) != "8UC1") return false;
    vector<double> hist;
    int binCount = 256;
    GenHist(img, hist, binCount);

    histeqMap = vector<int>(256, 0);
    for(int i = 0; i < 256; i++){
        double acc = 0;
        //the cumulative probability before i
        //because hist is a vector of double, we need to use 0.0 here!!
        acc = accumulate(hist.begin(), hist.begin()+i, 0.0);
        //map it to the scale of [0, 255]
        acc *= 255;
        acc = min(max((int)acc, 0), 255); 
        histeqMap[i] = (int)acc;
    }

    return true;
}

bool Histeq(cv::Mat& img){
    //p.93
    vector<int> histeqMap;

    if(!GetHisteqMap(img, histeqMap)) return false;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            img.at<uchar>(i, j) = histeqMap[img.at<uchar>(i, j)];
        }
    }
    return true;
};

void GetInverseEqHist(vector<double>& hist, vector<int>& inverseHistEqMap){
    //input hist is the normalized histogram
    //initialize as -1!
    inverseHistEqMap = vector<int>(256, -1);

    //find the inverse of histeqmap
    for(int i = 0; i < 256; i++){
        double acc = accumulate(hist.begin(), hist.begin()+i, 0.0);
        inverseHistEqMap[round(acc*255)] = i;
    }

    // for(int i = 0; i < 256; i++){
    //     cout << inverseHistEqMap[i] << " ";
    // }
    // cout << endl;

    //make sure every element in the domain [0,255] is meaningful
    int i = 0, j = 0;
    while(i < 256){
        while((i+1 < 256) && inverseHistEqMap[i+1] != -1){
            i++;
        }
        //now we find an "i" s.t. map[i] is meaningful but map[i+1] is not
        for(j = 1; (i+j < 256) && inverseHistEqMap[i+j] == -1; j++){
            //fill all meaningless map[i+1...?] as map[i]
            inverseHistEqMap[i+j] = inverseHistEqMap[i];
        }
        i += j;
    }

    // for(int i = 0; i < 256; i++){
    //     cout << inverseHistEqMap[i] << " ";
    // }
    // cout << endl;
};

bool Histst(cv::Mat& img, vector<double> stdHist){
    //p.98
    vector<int> histeqMap;
    
    if(!GetHisteqMap(img, histeqMap)) return false;

    //find the inverse of histogram equalization map for stdHist
    vector<int> inverseHistEqMap;
    GetInverseEqHist(stdHist, inverseHistEqMap);

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            int val = img.at<uchar>(i, j);
            //doing histogram equalization
            val = histeqMap[val];
            //from equalized histogram to target histogram
            val = inverseHistEqMap[val];
            val = min(max((int)val, 0), 255); 
            img.at<uchar>(i, j) = val;
        }
    }
    
    return true;
};

bool Histst(cv::Mat& img, cv::Mat& stdImg){
    //p.99
    vector<double> stdHist;
    GenHist(stdImg, stdHist);
    return Histst(img, stdHist);
};

void DrawHist(vector<double>& hist, cv::Mat& histImage, int img_h){
    //output image size
    int img_w = img_h;
    int bin_w = (int)((double)img_w/hist.size());

    histImage = cv::Mat(img_h, img_w, CV_8UC1, cv::Scalar(0));

    /// Normalize the result to [ 0, histImage.rows ]
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

    /// Draw for each channel
    for( int i = 1; i < hist.size(); i++ )
    {
        cv::line(histImage, cv::Point(bin_w*(i-1), img_h - hist[i-1]) ,
                        cv::Point(bin_w*(i), img_h - hist[i]),
                        cv::Scalar(255), 2, 8, 0);
    }
};

#ifdef CH3
int main(){
    bool isSave = false;
    cv::Mat img = cv::imread("images/Lenna.png", 0);
    cv::Mat work = img.clone();

    // Threshold
    cout << "Please input the threshold..." << endl;
    int threshold;
    cin >> threshold;
    work = img.clone();
    Threshold(work, threshold);
    // Show(work, "Threshold", isSave);
    vector<cv::Mat> thresholdImgs = {img, work};
    ShowHorizontal(thresholdImgs, string("Threshold") + "_" + to_string(threshold), isSave);

    //LinTran
    cout << "Please input dFa and dFb for linear transform..." << endl;
    double dFa, dFb;
    cin >> dFa >> dFb;
    work = img.clone();
    LinTran(work, dFa, dFb);
    // Show(work, "Linear Transform", isSave);
    vector<cv::Mat> linTranImgs = {img, work};
    string linTranTitle = string("Linear Transform") + " " + to_string_with_precision(dFa, 2) + " " + to_string_with_precision(dFb, 2);
    ShowHorizontal(linTranImgs, linTranTitle, isSave);

    //GammaTran
    cout << "Please input gamma and comp for gamma transform..." << endl;
    double gamma, comp;
    cin >> gamma >> comp;
    work = img.clone();
    GammaTran(work, gamma, comp);
    // Show(work, "Gamma Transform", isSave);
    vector<cv::Mat> gammaTranImgs = {img, work};
    string gammaTranTitle = string("Gamma Transform") + " " + to_string_with_precision(gamma, 2) + " " + to_string_with_precision(comp, 2);
    ShowHorizontal(gammaTranImgs, gammaTranTitle, isSave);

    //LogTran
    cout << "Please input dC for log transform..." << endl;
    double dC;
    cin >> dC;
    work = img.clone();
    LogTran(work, dC);
    // Show(work, "Log Transform", isSave);
    vector<cv::Mat> logTranImgs = {img, work};
    string logTranTitle = string("Log Transform") + " " + to_string_with_precision(dC, 2);
    ShowHorizontal(logTranImgs, "Log Transform", isSave);

    //ParLinTran
    cout << "Please input x1, x2, y1, y2 for partial linear transform..." << endl;
    int x1, x2, y1, y2;
    cin >> x1 >> x2 >> y1 >> y2;
    work = img.clone();
    ParLinTran(work, x1, x2, y1, y2);
    // Show(work, "Paritial Linear Transform", isSave);
    vector<cv::Mat> parLinTranImgs = {img, work};
    string parLinTranTitle = string("Paritial Linear Transform") + " " + to_string(x1) + " " + to_string(x2) + " " + to_string(y1) + " " + to_string(y2);
    ShowHorizontal(parLinTranImgs, parLinTranTitle, isSave);
    
    //Histogram equalization
    vector<double> hist;
    cout << "Please input the bin count of histogram for histogram equalization..." << endl;
    int n;
    cin >> n;
    work = img.clone();
    GenHist(work, hist, n);
    // ShowHist(hist, isSave);
    cv::Mat histImage;
    DrawHist(hist, histImage, img.rows);
    Histeq(work);
    // Show(img, "Original", isSave);
    // Show(work, "Histogram Equalization", isSave);
    vector<cv::Mat> HistEqImgs = {img, histImage, work};
    string histEqTitle = string("Histogram Equalization") + " " + to_string(n);
    ShowHorizontal(HistEqImgs, histEqTitle, isSave);

    //Histogram matching
    cv::Mat img_dark = cv::imread("images/dark.jfif", 0);
    cv::Mat img_light = cv::imread("images/light.jfif", 0);
    work = img.clone();
    Histst(work, img_dark);
    // Show(img, "Original", isSave);
    // Show(img_dark, "Dark Standard", isSave);
    // Show(work, "Histogram Matching to Dark", isSave);
    vector<cv::Mat> darkStdImgs = {img, img_dark, work};
    ShowHorizontal(darkStdImgs, "Histogram Matching to Dark", isSave);

    work = img.clone();
    Histst(work, img_light);
    // Show(img_light, "Light Standard", isSave);
    // Show(work, "Histogram Matching to Light", isSave);
    vector<cv::Mat> lightStdImgs = {img, img_light, work};
    ShowHorizontal(lightStdImgs, "Histogram Matching to Light", isSave);

    return 0;
}
#endif
