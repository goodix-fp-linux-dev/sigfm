#include <iostream>
#include <numeric>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <string>

std::string folder_path = "/home/mpi3d/Documents/sigfm-cpp/fingerprints/";
std::string ext = ".png";
auto number = 100;
auto finger_1 = 0;
auto finger_2 = 1;
auto distance_match = 0.75;
auto length_match = 0.5;
auto angle_match = 0.05;

auto clear = cv::imread(folder_path + "clear" + ext, cv::IMREAD_GRAYSCALE);

int compare(cv::Mat image_1, cv::Mat image_2)
{

	image_1 = 256 - clear + image_1;
	cv::normalize(image_1, image_1, 255, 0, cv::NORM_MINMAX, CV_8U);

	image_2 = 256 - clear + image_2;
	cv::normalize(image_2, image_2, 255, 0, cv::NORM_MINMAX, CV_8U);

	auto sift = cv::SIFT::create();

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat descriptors_1, descriptors_2;
	sift->detectAndCompute(image_1, cv::noArray(), keypoints_1, descriptors_1);
	sift->detectAndCompute(image_2, cv::noArray(), keypoints_2, descriptors_2);

	std::vector<std::vector<cv::DMatch>> matches_in;
	cv::BFMatcher::create()->knnMatch(descriptors_1, descriptors_2,
									  matches_in, 2);

	std::vector<std::pair<cv::Point2f, cv::Point2f>> matches_out;
	for (auto match_in : matches_in)
		if (match_in[0].distance < distance_match * match_in[1].distance)
		{
			auto match_out = std::make_pair(
				keypoints_1[match_in[0].queryIdx].pt,
				keypoints_2[match_in[0].trainIdx].pt);

			auto end = matches_out.end();
			if (std::find(matches_out.begin(), end, match_out) == end)
				matches_out.push_back(match_out);
		}

	std::vector<double> angles;
	auto length = matches_out.size();
	for (auto i = 0; i < length; i++)
	{
		auto match_1 = matches_out[i];
		for (auto j = i + 1; j < length; j++)
		{
			auto match_2 = matches_out[j];

			auto vector_1 = std::make_pair(
				match_1.first.x - match_2.first.x,
				match_1.first.y - match_2.first.y);
			auto vector_2 = std::make_pair(
				match_1.second.x - match_2.second.x,
				match_1.second.y - match_2.second.y);

			auto length_1 = sqrt(pow(vector_1.first, 2) +
								 pow(vector_1.second, 2));
			auto length_2 = sqrt(pow(vector_2.first, 2) +
								 pow(vector_2.second, 2));

			if (std::abs(length_1 - length_2) < length_match)
				angles.push_back(atan2(vector_1.first * vector_2.second -
										   vector_1.second * vector_2.first,
									   vector_1.first * vector_2.first +
										   vector_1.second * vector_2.second));
		}
	}

	auto max_count = 0;

	length = angles.size();
	for (auto i = 0; i < length; i++)
	{
		auto count = 0;
		auto angle_1 = angles[i];

		for (auto j = 0; j < length; j++)
		{
			if (i == j)
				continue;

			auto angle_2 = angles[j];

			auto distance = std::abs(angle_1 - angle_2);
			if (distance < angle_match or 2 * M_PI - distance < angle_match)
				count++;
		}
		if (count > max_count)
			max_count = count;
	}

	return max_count;
}

int main()
{
	if (clear.empty())
		return -1;

	std::vector<int> results;
	for (auto number_1 = 0; number_1 < number; number_1++)
	{
		for (auto number_2 = 0; number_2 < number; number_2++)
		{
			if (finger_1 == finger_2 && number_1 == number_2)
				continue;

			auto image_1 = cv::imread(folder_path + "finger-" +
										  std::to_string(finger_1) + "/" +
										  std::to_string(number_1) + ext,
									  cv::IMREAD_GRAYSCALE);

			if (image_1.empty())
				return -1;

			auto image_2 = cv::imread(folder_path + "finger-" +
										  std::to_string(finger_2) + "/" +
										  std::to_string(number_2) + ext,
									  cv::IMREAD_GRAYSCALE);

			if (image_2.empty())
				continue;

			results.push_back(compare(image_1, image_2));
		}
	}

	auto count = results.size();
	auto begin = results.begin();
	auto end = results.end();
	std::cout << "Count: " << count << std::endl;
	std::cout << "Mean: " << std::reduce(begin, end) / count << std::endl;
	std::cout << "Max: " << *std::max_element(begin, end) << std::endl;
	std::cout << "Min: " << *std::min_element(begin, end) << std::endl;
}