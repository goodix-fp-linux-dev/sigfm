#include <cstddef>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <ostream>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/file.h>
#include <string.h>
#include <cmath>
#include <tuple>
#include <vector>
#include <set>

using std::cout; 
using std::endl;
using cv::getTrackbarPos;
using cv::imread;
using std::max;
using std::min;
using std::to_string;
using std::set;
using std::fstream;
using std::string;
using std::vector;

// int finger1=0;
// int finger2=0;
// int img1=0;
// int img2=1;
// int dmatch=75;
// int mmatch=5;
// int lmatch=50;
// int amatch=50;


struct match{
	cv::Point2f p1;
	cv::Point2f p2;
	match(cv::Point2f ip1, cv::Point2f ip2){
		this->p1 = ip1;
		this->p2 = ip2;
	}
	match(){
		this->p1 = cv::Point2f(0,0);
		this->p2 = cv::Point2f(0,0);
	}
	bool operator== (const match& right) const {
		return std::tie(this->p1, this->p2)==std::tie(right.p1, right.p2);
	}
	bool operator< (const match& right) const {
		return this->p1.x < right.p1.x || this->p1.y < right.p1.y;
	}
};

struct angle{
	double cos;
	double sin;
	match corr_matches[2];
	angle(double cos, double sin, match m1, match m2){
		this->cos = cos;
		this->sin = sin;
		this->corr_matches[0] = m1;
		this->corr_matches[1] = m2;
	}
};

string folder_path = "/home/mango/fpr/";
string ext         = ".jpg";

cv::Mat clear = cv::imread(folder_path + "clear" + ext, 0);

void update(int, void *){
	int finger_1 = getTrackbarPos("finger", "image 1");
	cout << finger_1 << endl;
	int number_1 = getTrackbarPos("image", "image 1");
	cout << number_1 << endl;
	cv::Mat image_1 = imread(folder_path + "finger-" + to_string(finger_1) + "/" +
								to_string(number_1) + ext,
							0);

	if (image_1.empty())
		return;

	int finger_2 = getTrackbarPos("finger", "image 2");
	cout << finger_2 << endl;
	int number_2 = getTrackbarPos("image", "image 2");
	cout << number_2 << endl;
	cv::Mat image_2 = imread(folder_path + "finger-" + to_string(finger_2) + "/" +
								to_string(number_2) + ext,
							0);

	if (image_2.empty())
		return;

	if (finger_1 == finger_2 && number_1 == number_2)
		return;

	image_1 -= (256 - clear);
	cv::Mat ROI = cv::Mat::ones(cv::Size(image_1.size[0], image_1.size[1]), 0);

	unsigned char maximum = 0;
	unsigned char minimum = 255;
	image_1.forEach<unsigned char>(
		[&maximum, &minimum](unsigned char &p, const int *position) -> void {
			if (maximum < p) {
			maximum = p;
			}
			if (minimum > p) {
			minimum = p;
			}
		});

	int tmp = 255 / (maximum - minimum);
	image_1 = tmp * image_1 - minimum * tmp;

	image_2 -= (256 - clear);
	maximum = 0;
	minimum = 255;
	image_2.forEach<uchar>(
		[&maximum, &minimum](uchar &p, const int *position) -> void {
			if (maximum < p) {
			maximum = p;
			}
			if (minimum > p) {
			minimum = p;
			}
		});

	tmp = 255 / (maximum - minimum);
	image_2 = tmp * image_2 - minimum * tmp;

	cv::imshow("image 1", image_1);
	cv::imshow("image 2", image_2);

	vector<cv::KeyPoint> keypoints_1;
	vector<cv::KeyPoint> keypoints_2;
	cv::Mat descriptors_1;
	cv::Mat descriptors_2;

	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	sift->detectAndCompute(image_1, ROI, keypoints_1, descriptors_1);
	sift->detectAndCompute(image_2, ROI, keypoints_2, descriptors_2);

	float dist_match = (float)getTrackbarPos("distance match", "match") / 100;

	vector<vector<cv::DMatch>> points;
	cv::BFMatcher::create()->knnMatch(descriptors_1, descriptors_2, points, 2);

	vector<match> matches;
	for (int i = 0; i < points.size(); i++) {
		cv::DMatch match_1 = points[i][0];
		if (match_1.distance < dist_match * points[i][1].distance) {
			cout << "size:" << matches.size() << "/" << matches.max_size() << endl;
			matches.push_back({ keypoints_1[match_1.queryIdx].pt,
								keypoints_2[match_1.trainIdx].pt});
		}
	}


	vector<angle> angles;
	vector<vector<match>> angles_corresp_matches;
	// cout << angles_corresp_matches.max_size() << endl;

	float len_match = (float) getTrackbarPos("length match", "match") / 1000;

	set<match> set(matches.begin(), matches.end());
    matches.assign(set.begin(), set.end());
	cout << "size_after_shrink:" << matches.size() << "/" << matches.max_size() << endl;

	for(int j = 0; j < matches.size(); j++){
		match match_1 = matches[j];
		for(int k = j+1; k < matches.size(); k++){
			match match_2 = matches[k];

			double vec_1 [2] = {match_1.p1.x - match_2.p1.x,
									match_1.p1.y - match_2.p1.y};
			double vec_2 [2] = {match_1.p2.x - match_2.p2.x,
									match_1.p2.y - match_2.p2.y};

			double length_1 = sqrt(pow(vec_1[0],2) + pow(vec_1[1],2));
			double length_2 = sqrt(pow(vec_2[0],2) + pow(vec_2[1],2));

			if (1 - min(length_1, length_2) / 
					max(length_1, length_2) <= len_match){
	
				double product = length_1 * length_2;
				double vec [2] = {};
				angles.push_back(angle(
					M_PI / 2 + asin((vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) / product),
					acos((vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]) / product),
					match_1,
					match_2
					));
				angles_corresp_matches.push_back({match_1, match_2});
			}
		}
	}

	int count = 0;
	int max_count = 0;
	float angle_match = (float) getTrackbarPos("angle match", "match") / 1000;
	vector<match> max_true_matches;

	for(int j = 0; j < angles.size(); j++){
		count = 0;
		angle angle_1 = angles[j];
		// match match_1[2] = {angles_corresp_matches[j][0], angles_corresp_matches[j][1]};
		vector<match> true_matches;

		for(int k = j+1; k < angles.size(); k++){

			angle angle_2 = angles[k];
			match match_2[2] = {angles_corresp_matches[k][0], angles_corresp_matches[k][1]};


			if (1 - min(angle_1.sin, angle_2.sin) / 
					max(angle_1.sin, angle_2.sin) <= angle_match &&
				1 - min(angle_1.cos, angle_2.cos) /
					max(angle_1.cos, angle_2.cos) <= angle_match){

				count += 1;
				
				for(match match_ : {angle_1.corr_matches[0], angle_1.corr_matches[1], angle_2.corr_matches[0], angle_2.corr_matches[0]}){
					bool innocence = true;
					for(match waround : true_matches){
						if(waround == match_){
							innocence = false;
							break;
						}
					}
					if(innocence){
						true_matches.push_back(match_);
					}
				}
			}
			if(count>=max_count){
				max_count = count;
				max_true_matches = true_matches;
			}
		}
	}

	cv::Mat image_3; cv::hconcat(image_1, image_2, image_3);
	cv::cvtColor(image_3, image_3, cv::COLOR_GRAY2RGB);
	for (int i = 0; i < matches.size(); i+=2){
		match match_ = matches[i];
		cv::Scalar color;
		bool innocence_1 = true, innocence_2 = true;
		for(match waround : max_true_matches){
			if(waround == match_){
				innocence_2 = false;
				break;
			}
		}
		color = innocence_1 && innocence_2 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

		cv::line(image_3, match_.p1, cv::Point2f(match_.p2.x+image_1.size().width, match_.p2.y), color, 1, cv::LINE_AA);
		cv::circle(image_3, match_.p1, 3, color, 1, cv::LINE_AA);
		cv::circle(image_3, cv::Point2f(match_.p2.x+image_1.size().width, match_.p2.y), 3, color, 1, cv::LINE_AA);
	}
	imshow("match", image_3);
}


int main(){
	cv::namedWindow("image 1", cv::WINDOW_NORMAL);
	cv::namedWindow("image 2", cv::WINDOW_NORMAL);
	cv::namedWindow("match", cv::WINDOW_NORMAL);

	cv::createTrackbar("finger", "image 1", NULL, 9, update);
	cv::createTrackbar("image", "image 1", NULL, 99, update);
	cv::createTrackbar("finger", "image 2", NULL, 9, update);
	cv::createTrackbar("image", "image 2", NULL, 99, update);
	cv::createTrackbar("distance match", "match", NULL, 100, update);
	cv::createTrackbar("min match", "match", NULL, 50, update);
	cv::createTrackbar("length match", "match", NULL, 1000, update);
	cv::createTrackbar("angle match", "match", NULL, 1000, update);

	update(0, nullptr);

	while(true){
		if((cv::waitKey() & 0xff) == 27)
			break;
	}

	cv::destroyAllWindows();
	return 0;

}