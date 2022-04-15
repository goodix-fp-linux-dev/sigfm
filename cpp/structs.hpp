#include <algorithm>
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
#include <string.h>
#include <cmath>
#include <sstream>
#include <tuple>
#include <vector>
#include <set>
#include <sys/file.h>
#include <string.h>

using std::min; using std::max;
using std::filesystem::recursive_directory_iterator;
using std::fstream;
using std::cout; 
using std::endl;
using cv::getTrackbarPos;
using cv::imread;
using std::to_string;
using std::set;
using std::fstream;
using std::string;
using std::vector;
namespace structs {
	struct match{
		cv::Point2i p1;
		cv::Point2i p2;
		match(cv::Point2i ip1, cv::Point2i ip2){
			this->p1 = ip1;
			this->p2 = ip2;
		}
		match(){
			this->p1 = cv::Point2i(0,0);
			this->p2 = cv::Point2i(0,0);
		}
		bool operator== (const match& right) const {
			return std::tie(this->p1, this->p2)==std::tie(right.p1, right.p2);
		}
		bool operator< (const match& right) const {
			return (this->p1.y < right.p1.y) || ((this->p1.y < right.p1.y) && this->p1.x < right.p1.x);
		}
	};
	inline std::ostream& operator<<(std::ostream& os, const match& arg){
		os << "Point 1: (" << arg.p1.x << ", " << arg.p1.y << ")" << endl \
		   << "Point 2: (" << arg.p2.x << ", " << arg.p2.y << ")" << endl;
		return os;
	} 
	inline std::string to_string(match const& arg)
	{
		std::ostringstream ss;
		ss << arg;
		return std::move(ss).str();  // enable efficiencies in c++17
	}
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
}