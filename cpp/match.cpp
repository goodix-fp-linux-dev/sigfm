#include <filesystem>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sys/file.h>
#include <string.h>
#include <math.h>
#include <tgmath.h>

using std::cout; using std::cin;
using std::endl; using std::string;
using std::vector; using std::to_string;
using std::min; using std::max;
using std::filesystem::recursive_directory_iterator;
using std::fstream;
 

int DISTANCE_MATCH = 0.75;
int LENGTH_MATCH = 0.05;
int ANGLE_MATCH = 0.05;
int MIN_MATCH = 5;
string ext = ".bin";

bool match(string img_path, string clear_path, string sample_path){

    cv::Mat image = (256 - cv::imread(clear_path)) - cv::imread(img_path);

    cv::Mat ROI = cv::Mat::ones(cv::Size(image.size[0], image.size[1]), 0);

    uchar maximum = 0; uchar minimum = 255;
    image.forEach<uchar>([&maximum, &minimum](uchar &p, const int * position) -> void {
        if (maximum < p){
            maximum = p;
        }
        if (minimum > p) {
            minimum = p;
        }
    });
    
    image = (255/(maximum-minimum)) * (image-minimum);
    vector<cv::KeyPoint> keypoints_image;
    cv::Mat descriptors_image; 
    cv::SIFT::create()->detectAndCompute(image, cv::noArray(), keypoints_image, descriptors_image);

    if(keypoints_image.size() < MIN_MATCH)
        return false;
    cv::Ptr<cv::BFMatcher> BFM = cv::BFMatcher::create();
    for(auto file : recursive_directory_iterator(sample_path)){
        if(((string)file.path()).substr(((string)file.path()).size()-4, 4) == ext){
            vector<cv::KeyPoint> keypoints_match;
            cv::Mat descriptors_match; 
            cv::FileStorage store(file.path(), cv::FileStorage::READ);

            cv::FileNode n1 = store["keypoints"];
            cv::FileNode n2 = store["descriptors"];
            
            cv::read(n1,keypoints_match);
            cv::read(n2,descriptors_match);
            
            store.release();
            
            if(keypoints_match.size() < MIN_MATCH){
                continue;
            }
            
            vector<vector<cv::DMatch>> points;
            BFM->knnMatch(descriptors_image, descriptors_match, points, 2);
            
            vector<cv::KeyPoint> matches;
            for (int i = 0; i < points.size(); i++){
                cv::DMatch match_1 = points[i][0];
                if(match_1.distance < DISTANCE_MATCH * points[i][1].distance){
                    matches.push_back((keypoints_image[match_1.queryIdx],
                                       keypoints_match[match_1.trainIdx]));
                }
            }
            cout << matches.size() << endl;
            if(matches.size() < MIN_MATCH){
                continue;
            }

            vector<vector<double>> angles;
            
            for(int j = 0; j < matches.size(); j++){
                cv::KeyPoint match_1 = matches[j];

                for(int k = j+1; k < matches.size(); k++){
                    cv::KeyPoint match_2 = matches[k];

                    vector vec_1 = {match_1.pt.x - match_2.pt.x,
                                    match_1.pt.y - match_2.pt.y};
                    vector vec_2 = {match_1.pt.x - match_2.pt.x,
                                    match_1.pt.y - match_2.pt.y};
                    double a = vec_1[0] * vec_1[0];
                    double length_1 = sqrt(vec_1[0] * vec_1[0] + vec_1[1] * vec_1[1]);
                    double length_2 = sqrt(vec_2[0] * vec_2[0] + vec_2[1] * vec_2[1]);

                    if (1 - min(length_1, length_2) / 
                            max(length_1, length_2) <= LENGTH_MATCH){
            
                        double product = length_1 * length_2;
                        vector<double> vec = {M_PI / 2 + asin((vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) / product),
                                                         acos((vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]) / product)};
                        angles.push_back(vec);
                    }
                }
            }

            if(angles.size()<MIN_MATCH)
                continue;

            int count = 0;
            for(int j = 0; j < angles.size(); j++){
                vector<double> angle_1 = angles[j];
                for(int k = j+1; k < angles.size(); k++){
                    vector<double> angle_2 = angles[k];

                    if (1 - min(angle_1[0], angle_2[0]) / 
                            max(angle_1[0], angle_2[0]) <= ANGLE_MATCH &&
                        1 - min(angle_1[1], angle_2[1]) /
                            max(angle_1[1], angle_2[1]) <= ANGLE_MATCH){
                            count += 1;

                        if (count >= MIN_MATCH)
                        return true;
                    }
                }
            }
            cout << count << endl;
        }
    }
    return false;
}

int main(int argc, char** argv){
    cout << match(argv[1], argv[2], argv[3]);
    return 0;
}