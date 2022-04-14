#include <filesystem>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sys/file.h>
#include <string.h>

using std::cout;
using std::endl;
using std::fstream;
using std::string;
using std::to_string;
using std::vector;
using std::filesystem::recursive_directory_iterator;

string ext = ".png";
string dir = "/home/mango/fpr";

void compute_finger(string images_path, string clear_path, string save_path) {
  cv::Mat clear = cv::imread(clear_path, 0);

  int i = 0;
  for (const auto file : recursive_directory_iterator(images_path)) {
    string path = file.path();

    if ((path.substr(path.size() - 4, 4) == ext) && path != clear_path) {
      cv::Mat image = (256 - clear) + cv::imread(path, 0);
      cv::Mat ROI = cv::Mat::ones(cv::Size(image.size[0], image.size[1]), 0);

      uchar maximum = 0;
      uchar minimum = 255;
      image.forEach<uchar>(
          [&maximum, &minimum](uchar &p, const int *position) -> void {
            if (maximum < p) {
              maximum = p;
            }
            if (minimum > p) {
              minimum = p;
            }
          });

      int tmp = 255 / (maximum - minimum);
      image = tmp * image - minimum * tmp;

      vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      cv::SIFT::create()->detectAndCompute(image, ROI, keypoints, descriptors);

      cv::FileStorage store(
          save_path + to_string(i) + ".bin",
          cv::FileStorage::WRITE);
      cv::write(store, "keypoints", keypoints);
      cv::write(store, "descriptors", descriptors);
      store.release();
      i++;
    }
  }
}

int main() {
  compute_finger("/home/mango/fpr",
                 "/home/mango/fpr/clear.jpg", "/home/mango/fpr");
}
