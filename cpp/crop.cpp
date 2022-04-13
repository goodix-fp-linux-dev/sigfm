#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <sys/file.h>
#include <string.h>


using std::string;
using std::filesystem::recursive_directory_iterator;
using std::vector;

int WIDTH = 64;
int HEIGHT = 80;

string ext = ".jpg";
string dir = "/home/mango/fpr";

cv::Rect ROI(0,0,WIDTH,HEIGHT);

int main(){
    vector<string> files;
    for (const auto file : recursive_directory_iterator(dir)){
        string path = (string) file.path();
        if(path.substr(path.size()-4, 4) == ext){
            cv::Mat img = cv::imread(path, 0);
            if(!(img.size[0] == HEIGHT && img.size[1] == WIDTH)){
                cv::Mat croppedImage = img(ROI).clone();
                cv::imwrite(path.substr(0, path.size()-4) + ext, croppedImage);
    }}}
    return 0;
}