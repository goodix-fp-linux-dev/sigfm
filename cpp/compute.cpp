#include "structs.hpp"


string ext = ".png";
string dir = "/home/mango/fpr";

void compute_finger(string images_path, string clear_path, string save_path) {
	cv::Mat clear = cv::imread(clear_path, 0);

	int i = 0;
	for (const auto file : recursive_directory_iterator(images_path)) {
		string path = file.path();

		if ((path.substr(path.size() - 4, 4) == ext) && path != clear_path) {
			cv::Mat image = clear - cv::imread(path, 0);

			cv::Mat ROI = cv::Mat::ones(cv::Size(image.size[0], image.size[1]), 0);

			double maximum;
			double minimum;

			minMaxLoc(image, &minimum, &maximum, NULL, NULL);

			double tmp = 255 / (maximum - minimum);
			image = tmp * (image - minimum);

			vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;

			cv::SIFT::create()->detectAndCompute(image, ROI, keypoints, descriptors);

			cv::FileStorage store(
					save_path + to_string(i) + ".yml",
					cv::FileStorage::WRITE);
			cv::write(store, "keypoints", keypoints);
			cv::write(store, "descriptors", descriptors);
			store.release();
			i++;
		}
	}
}

int main() {
	compute_finger("/home/mango/fpr", "/home/mango/fpr/clear.jpg", "/home/mango/fpr");
}
