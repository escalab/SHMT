#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp> //line
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/hal/interface.h> //CV_8UC3
#include <iostream>
#include <map>
#include <iomanip> //setw
#include "utility.h"

using namespace std;

std::string type2str(int type) {
  std::string r;
  
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  
  switch ( depth ) { 
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  
  r += "C";
  r += (chans+'0');
  
  return r;
};

void stringReplace(string& base, string from, string to){
    //start search from last end to speed up
    size_t pos = 0;
    while((pos = base.find(from, pos)) != string::npos){
        base.replace(pos, from.length(), to);
    }
};

void pad(cv::Mat& img, int padt, int padb, int padl, int padr, int mode){
    //https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5
    //mode could be cv::BORDER_CONSTANT, cv::BORDER_REPLICATE, ...
    int channels = img.channels();
    cv::Mat tmp;

    if(channels == 1){
        tmp = cv::Mat(cv::Size(img.cols+padl+padr, img.rows+padt+padb), CV_8UC1, cv::Scalar(0));
    }else if(channels == 3){
        tmp = cv::Mat(cv::Size(img.cols+padl+padr, img.rows+padt+padb), CV_8UC3, cv::Scalar(0));
    }

    if(mode == cv::BORDER_CONSTANT)
        copyMakeBorder(img, tmp, padt, padb, padl, padr, cv::BORDER_CONSTANT, cv::Scalar(0));
    else
        copyMakeBorder(img, tmp, padt, padb, padl, padr, mode);
    img = tmp;
};

void ConcatHorizontal(vector<cv::Mat>& imgs, cv::Mat& target){
    int channels = imgs[0].channels();
    int finalHeight = 0; //largest height
    for(int i = 0; i < imgs.size(); i++){
        finalHeight = max(finalHeight, imgs[i].rows);
    }

    if(channels == 1){
        target = cv::Mat(cv::Size(0, finalHeight), CV_8UC1, cv::Scalar(0));
    }else if(channels == 3){
        target = cv::Mat(cv::Size(0, finalHeight), CV_8UC3, cv::Scalar(0));
    }
    
    for(int i = 0; i < imgs.size(); i++){
        if(imgs[i].rows < finalHeight){
            //padding for imgs[i]
            pad(imgs[i], 0, finalHeight - imgs[i].rows, 0, 0);
        }
        hconcat(target, imgs[i], target);
    }
};

void Show(cv::Mat& img, string title, bool save){
    if(save){
        stringReplace(title, " ", "_");
        cv::imwrite("images/result/" + title + ".png", img);
    }else{
        cv::namedWindow( title, cv::WINDOW_AUTOSIZE);// Create a window for display.
        cv::imshow(title, img);                   // Show our image inside it.
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
};

void ShowHorizontal(vector<cv::Mat>& imgs, string title, bool save){
    cv::Mat target;
    ConcatHorizontal(imgs, target);

    if(save){
        stringReplace(title, " ", "_");
        cv::imwrite("images/result/" + title + ".png", target);
    }else{
        cv::namedWindow( title, cv::WINDOW_AUTOSIZE);// Create a window for display.
        cv::imshow(title, target);                   // Show our image inside it.
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
};

// template function's implementation should not be put in cpp file
// template <typename T>
// std::string to_string_with_precision(const T a_value, const int n)
// {
//     std::ostringstream out;
//     out.precision(n);
//     out << std::fixed << a_value;
//     return out.str();
// };

//from CH4
void PrintVector(vector<double>& row){
    for(int i = 0; i < row.size(); i++){
        cout << setw(10) << fixed << setprecision(5) << row[i];
    }
    cout << endl;
};

//from CH4
void PrintMatrix(vector<vector<double>>& matrix){
    for(int i = 0; i < matrix.size(); i++){
        PrintVector(matrix[i]);
    }
    cout << endl;
}

//from CH4
void SwapRow(vector<double>& row1, vector<double>& row2){
    int n = row1.size();
    for(int i = 0; i < n; i++){
        swap(row1[i], row2[i]);
    }
}

//from CH4
void MultiplyRow(vector<double>& row, double ratio){
    for(int i = 0; i < row.size(); i++){
        row[i] *= ratio;
    }
}

//from CH4
void MultiplySubstractRow(vector<double>& row1, vector<double>& row2, double ratio){
    //row1 = row1 - row2 * ratio
    for(int i = 0; i < row1.size(); i++){
        row1[i] -= row2[i] * ratio;
    }
}

//from CH4
bool InvMat(vector<vector<double>>& matrix){
    //https://ccjou.wordpress.com/2013/02/20/%E9%AB%98%E6%96%AF%E6%B6%88%E5%8E%BB%E6%B3%95/
    //https://www.geeksforgeeks.org/gaussian-elimination/
    //https://www.geeksforgeeks.org/program-for-gauss-jordan-elimination-method/
    //https://www.geeksforgeeks.org/finding-inverse-of-a-matrix-using-gauss-jordan-method/
    int n = matrix.size();

    //create augmented matrix
    vector<vector<double>> augMatrix = vector<vector<double>>(n, vector<double>(2 * n, 0.0));
    for(int i = 0; i < n; i++){
        //copy from input matrix
        for(int j = 0; j < n; j++){
            augMatrix[i][j] = matrix[i][j];
        }
        //the right part is identity matrix
        for(int j = n; j < 2*n; j++){
            //elements on diagonal are 1, others are 0
            augMatrix[i][j] = (int)(j-n == i);
        }
    }

    for(int row = 0; row < n; row++){
        //ensure the diagonal of augmented matrix's left part contains no 0
        //?
        if(augMatrix[row][row] == 0){
            int row2;
            for(row2 = row+1; row2 < n; row2++){
                if(augMatrix[row2][row] != 0){
                    //swap row i and row j
                    SwapRow(augMatrix[row], augMatrix[row2]);
                    break;
                }
            }
            //the column is all 0
            if(row2 == n) return false;
        }
        
        //R[row2] = R[row2] - ratio * R[row]
        for(int row2 = 0; row2 < n; row2++){
            //don't operate with itself!
            if(row == row2)continue;
            //make the column [?][row] all zero besides [row][row]
            double ratio = augMatrix[row2][row]/augMatrix[row][row];
            MultiplySubstractRow(augMatrix[row2], augMatrix[row], ratio);
        }
        
        //make diagonal all 1
        if(augMatrix[row][row] != 1){
            double ratio = 1.0/augMatrix[row][row];
            MultiplyRow(augMatrix[row], ratio);
        }
    }

    // //R[j] = R[j] - k * R[i]
    // for(int row1 = 0; row1 < n; row1++){
    //     for(int row2 = 0; row2 < n; row2++){
    //         //don't operate with itself!
    //         if(row1 == row2)continue;
    //         //make column row1 all zero besides [row1][row1]
    //         double ratio = augMatrix[row2][row1]/augMatrix[row1][row1];
    //         // augMatrix[row2] = augMatrix[row2] - ratio * augMatrix[row1];
    //         for(int col = 0; col < 2*n; col++){
    //             augMatrix[row2][col] -= ratio * augMatrix[row1][k];
    //         }
    //     }
    // }

    // //make diagonal all 1
    // for(int row = 0; row < n; row++){
    //     if(augMatrix[row][row] != 1){
    //         double ratio = 1.0/augMatrix[row][row];
    //         for(int col = 0; col < 2*n; col++){
    //             augMatrix[row][col] *= ratio;
    //         }
    //     }
    // }

    //copy back to input matrix
    for(int row = 0; row < n; row++){
        for(int col = 0; col < n; col++){
            matrix[row][col] = augMatrix[row][col+n];
        }
    }

    return true;
}

//from CH4
bool ProdMat(vector<vector<double>>& mat1, vector<vector<double>>& mat2, vector<vector<double>>& res){
    int m = mat1.size(), n = mat1[0].size(), p = mat2[0].size();
    if(n != mat2.size()) return false;

    res = vector<vector<double>>(m, vector<double>(p, 0.0));

    for(int i = 0; i < m; i++){
        for(int j = 0; j < p; j++){
            double val = 0.0;
            for(int k = 0; k < n; k++){
                val += mat1[i][k] * mat2[k][j];
            }
            res[i][j] = val;
        }
    }

    return true;
};
