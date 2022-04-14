#include "structs.hpp"

 

double DISTANCE_MATCH = 0.75;
double LENGTH_MATCH = 0.05;
double ANGLE_MATCH = 0.05;
double MIN_MATCH = 5;
string ext = ".bin";

bool fingerprint_match(string img_path, string clear_path, string sample_path){

	cv::Mat image = (256 - cv::imread(clear_path)) - cv::imread(img_path);

	cv::Mat ROI = cv::Mat::ones(cv::Size(image.size[0], image.size[1]), 0);

	double maximum;
	double minimum;

	minMaxLoc(image, &minimum, &maximum, NULL, NULL);
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
			
			vector<match> matches;
			for (int i = 0; i < points.size(); i++) {
				cv::DMatch match_1 = points[i][0];
				if (match_1.distance < DISTANCE_MATCH * points[i][1].distance) {
					matches.push_back({ keypoints_image[match_1.queryIdx].pt,
										keypoints_match[match_1.trainIdx].pt});
				}
			}
			cout << matches.size() << endl;
			if(matches.size() < MIN_MATCH){
				continue;
			}

			vector<angle> angles;

			set<match> set(matches.begin(), matches.end());
			matches.assign(set.begin(), set.end());

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
							max(length_1, length_2) <= LENGTH_MATCH){
			
						double product = length_1 * length_2;
						double vec [2] = {};
						angles.push_back(angle(
							M_PI / 2 + asin((vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) / product),
							acos((vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]) / product),
							match_1,
							match_2
							));
					}
				}
			}

			if(angles.size()<MIN_MATCH)
				continue;

			int count = 0;
			for(int j = 0; j < angles.size(); j++){
				angle angle_1 = angles[j];
				for(int k = j+1; k < angles.size(); k++){
					angle angle_2 = angles[k];

					if (1 - min(angle_1.sin, angle_2.sin) / 
							max(angle_1.sin, angle_2.sin) <= ANGLE_MATCH &&
						1 - min(angle_1.cos, angle_2.cos) /
							max(angle_1.cos, angle_2.cos) <= ANGLE_MATCH){

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
	cout << fingerprint_match(argv[1], argv[2], argv[3]);
	return 0;
}