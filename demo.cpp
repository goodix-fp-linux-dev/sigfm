#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <string>

std::string folder_path = "/home/mpi3d/Documents/sigfm-cpp/fingerprints/";
std::string ext = ".png";

auto clear = cv::imread(folder_path + "clear" + ext, cv::IMREAD_GRAYSCALE);

// TODO 12 bit depth

void update(int, void *)
{
	auto finger_1 = cv::getTrackbarPos("finger", "image 1");
	auto number_1 = cv::getTrackbarPos("image", "image 1");

	auto finger_2 = cv::getTrackbarPos("finger", "image 2");
	auto number_2 = cv::getTrackbarPos("image", "image 2");

	if (finger_1 == finger_2 && number_1 == number_2)
		return;

	auto image_1 = cv::imread(folder_path + "finger-" +
								  std::to_string(finger_1) + "/" +
								  std::to_string(number_1) + ext,
							  cv::IMREAD_GRAYSCALE);

	if (image_1.data == NULL)
		return;

	auto image_2 = cv::imread(folder_path + "finger-" +
								  std::to_string(finger_2) + "/" +
								  std::to_string(number_2) + ext,
							  cv::IMREAD_GRAYSCALE);

	if (image_2.data == NULL)
		return;

	image_1 = 256 - clear + image_1;
	cv::normalize(image_1, image_1, 255, 0, cv::NORM_MINMAX, CV_8U);

	image_2 = 256 - clear + image_2;
	cv::normalize(image_2, image_2, 255, 0, cv::NORM_MINMAX, CV_8U);

	cv::imshow("image 1", image_1);
	cv::imshow("image 2", image_2);

	auto sift = cv::SIFT::create();

	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat descriptors_1, descriptors_2;
	sift->detectAndCompute(image_1, cv::noArray(), keypoints_1, descriptors_1);
	sift->detectAndCompute(image_2, cv::noArray(), keypoints_2, descriptors_2);

	auto distance_match =
		(double)cv::getTrackbarPos("distance match", "match") / 100;

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

	std::cout << "Count 1: " << matches_out.size() << std::endl; // TODO Remove

	auto length_match =
		(double)cv::getTrackbarPos("length match", "match") / 100;

	std::vector<std::pair<double, std::pair<
									  std::pair<cv::Point2f, cv::Point2f>,
									  std::pair<cv::Point2f, cv::Point2f>>>>
		angles;
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
				angles.push_back(std::make_pair(
					atan2(vector_1.first * vector_2.second -
							  vector_1.second * vector_2.first,
						  vector_1.first * vector_2.first +
							  vector_1.second * vector_2.second),
					std::make_pair(match_1, match_2)));
		}
	}

	std::cout << "Count 2: " << angles.size() << std::endl; // TODO Remove

	auto max_count = 0;
	auto angle_match =
		(double)cv::getTrackbarPos("angle match", "match") / 100;
	std::vector<std::pair<cv::Point2f, cv::Point2f>> true_matches,
		max_true_matches;

	length = angles.size();
	for (auto i = 0; i < length; i++)
	{
		auto count = 0;
		auto angle_1 = angles[i];
		true_matches = {angle_1.second.first, angle_1.second.second};

		for (auto j = 0; j < length; j++)
		{
			if (i == j)
				continue;

			auto angle_2 = angles[j];

			auto distance = std::abs(angle_1.first - angle_2.first);
			if (distance < angle_match or 2 * M_PI - distance < angle_match)
			{
				count++;
				true_matches.push_back(angle_2.second.first);
				true_matches.push_back(angle_2.second.second);
			}
		}
		if (count > max_count)
		{
			max_count = count;
			max_true_matches = true_matches;
		}
	}

	std::cout << max_count << std::endl; // TODO Remove
	for (auto match : max_true_matches)
		std::cout << "[(" << match.first.x << ", " << match.first.y << "), ("
				  << match.second.x << ", " << match.first.y << ")]"
				  << std::endl; // TODO Remove

	cv::Mat image_3;
	cv::hconcat(image_1, image_2, image_3);
	cv::cvtColor(image_3, image_3, cv::COLOR_GRAY2RGB);

	for (auto match : matches_out)
	{
		auto end = max_true_matches.end();
		if (std::find(max_true_matches.begin(), end, match) == end)
		{
			match.second.x += image_1.cols;
			cv::line(image_3, match.first, match.second, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
			cv::circle(image_3, match.first, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
			cv::circle(image_3, match.second, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
		}
	}

	for (auto match : max_true_matches)
	{
		match.second.x += image_1.cols;
		cv::line(image_3, match.first, match.second, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
		cv::circle(image_3, match.first, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
		cv::circle(image_3, match.second, 3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
	}

	cv::Scalar color;
	if (max_count > cv::getTrackbarPos("min match", "match"))
		color = cv::Scalar(0, 255, 0);
	else
		color = cv::Scalar(0, 0, 255);

	cv::rectangle(image_3, cv::Point(0, 0),
				  cv::Point(image_3.cols - 1, image_3.rows - 1), color, 1);

	cv::imshow("match", image_3);
}

int main()
{
	if (clear.data == NULL)
		return -1;

	cv::namedWindow("image 1", cv::WINDOW_NORMAL);
	cv::namedWindow("image 2", cv::WINDOW_NORMAL);
	cv::namedWindow("match", cv::WINDOW_NORMAL);

	cv::createTrackbar("finger", "image 1", NULL, 9, update);
	cv::createTrackbar("image", "image 1", NULL, 99, update);
	cv::createTrackbar("finger", "image 2", NULL, 9, update);
	cv::createTrackbar("image", "image 2", NULL, 99, update);
	cv::createTrackbar("distance match", "match", NULL, 100, update);
	cv::createTrackbar("length match", "match", NULL, 100, update);
	cv::createTrackbar("angle match", "match", NULL, 100, update);
	cv::createTrackbar("min match", "match", NULL, 50, update);

	cv::setTrackbarPos("image", "image 1", 0);
	cv::setTrackbarPos("image", "image 2", 1);
	cv::setTrackbarPos("distance match", "match", 75);
	cv::setTrackbarPos("length match", "match", 50);
	cv::setTrackbarPos("angle match", "match", 5);
	cv::setTrackbarPos("min match", "match", 5);

	update(0, nullptr);

	while (true)
		if ((cv::waitKey() & 0xff) == 27)
			break;

	cv::destroyAllWindows();
	return 0;
}