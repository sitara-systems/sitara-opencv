#include "OpenCVHelpers.h"

namespace sitara {
	namespace opencv {
		#define mkcase(x, y) {case x: return y;}

		int getTargetChannelsFromCode(int conversionCode) {
			switch (conversionCode) {
				mkcase(cv::COLOR_RGB2RGBA, 4)	mkcase(cv::COLOR_RGBA2RGB, 3) mkcase(cv::COLOR_RGB2BGRA, 4)
					mkcase(cv::COLOR_RGBA2BGR, 3) mkcase(cv::COLOR_BGR2RGB, 3) mkcase(cv::COLOR_BGRA2RGBA, 4)
					mkcase(cv::COLOR_BGR2GRAY, 1) mkcase(cv::COLOR_RGB2GRAY, 1) mkcase(cv::COLOR_GRAY2RGB, 3)
					mkcase(cv::COLOR_GRAY2RGBA, 4) mkcase(cv::COLOR_BGRA2GRAY, 1) mkcase(cv::COLOR_RGBA2GRAY, 1)
					mkcase(cv::COLOR_BGR5652BGR, 3) mkcase(cv::COLOR_BGR5652RGB, 3) mkcase(cv::COLOR_BGR5652BGRA, 4)
					mkcase(cv::COLOR_BGR5652RGBA, 4) mkcase(cv::COLOR_BGR5652GRAY, 1) mkcase(cv::COLOR_BGR5552BGR, 3)
					mkcase(cv::COLOR_BGR5552RGB, 3) mkcase(cv::COLOR_BGR5552BGRA, 4) mkcase(cv::COLOR_BGR5552RGBA, 4)
					mkcase(cv::COLOR_BGR5552GRAY, 1) mkcase(cv::COLOR_BGR2XYZ, 3) mkcase(cv::COLOR_RGB2XYZ, 3)
					mkcase(cv::COLOR_XYZ2BGR, 3) mkcase(cv::COLOR_XYZ2RGB, 3) mkcase(cv::COLOR_BGR2YCrCb, 3)
					mkcase(cv::COLOR_RGB2YCrCb, 3) mkcase(cv::COLOR_YCrCb2BGR, 3) mkcase(cv::COLOR_YCrCb2RGB, 3)
					mkcase(cv::COLOR_BGR2HSV, 3) mkcase(cv::COLOR_RGB2HSV, 3) mkcase(cv::COLOR_BGR2Lab, 3)
					mkcase(cv::COLOR_RGB2Lab, 3) mkcase(cv::COLOR_BayerGB2BGR, 3) mkcase(cv::COLOR_BayerBG2RGB, 3)
					mkcase(cv::COLOR_BayerGB2RGB, 3) mkcase(cv::COLOR_BayerRG2RGB, 3) mkcase(cv::COLOR_BGR2Luv, 3)
					mkcase(cv::COLOR_RGB2Luv, 3) mkcase(cv::COLOR_BGR2HLS, 3) mkcase(cv::COLOR_RGB2HLS, 3)
					mkcase(cv::COLOR_HSV2BGR, 3) mkcase(cv::COLOR_HSV2RGB, 3) mkcase(cv::COLOR_Lab2BGR, 3)
					mkcase(cv::COLOR_Lab2RGB, 3) mkcase(cv::COLOR_Luv2BGR, 3) mkcase(cv::COLOR_Luv2RGB, 3)
					mkcase(cv::COLOR_HLS2BGR, 3) mkcase(cv::COLOR_HLS2RGB, 3) mkcase(cv::COLOR_BayerBG2RGB_VNG, 3)
					mkcase(cv::COLOR_BayerGB2RGB_VNG, 3) mkcase(cv::COLOR_BayerRG2RGB_VNG, 3)
					mkcase(cv::COLOR_BayerGR2RGB_VNG, 3) mkcase(cv::COLOR_BGR2HSV_FULL, 3)
					mkcase(cv::COLOR_RGB2HSV_FULL, 3) mkcase(cv::COLOR_BGR2HLS_FULL, 3)
					mkcase(cv::COLOR_RGB2HLS_FULL, 3) mkcase(cv::COLOR_HSV2BGR_FULL, 3)
					mkcase(cv::COLOR_HSV2RGB_FULL, 3) mkcase(cv::COLOR_HLS2BGR_FULL, 3)
					mkcase(cv::COLOR_HLS2RGB_FULL, 3) mkcase(cv::COLOR_LBGR2Lab, 3) mkcase(cv::COLOR_LRGB2Lab, 3)
					mkcase(cv::COLOR_LBGR2Luv, 3) mkcase(cv::COLOR_LRGB2Luv, 3) mkcase(cv::COLOR_Lab2LBGR, 4)
					mkcase(cv::COLOR_Lab2LRGB, 4) mkcase(cv::COLOR_Luv2LBGR, 4) mkcase(cv::COLOR_Luv2LRGB, 4)
					mkcase(cv::COLOR_BGR2YUV, 3) mkcase(cv::COLOR_RGB2YUV, 3) mkcase(cv::COLOR_YUV2BGR, 3)
					mkcase(cv::COLOR_YUV2RGB, 3)
			default: return 0;
			}
		}

		float getMaxVal(int cvDepth) {
			switch (cvDepth) {
			case CV_8U: return std::numeric_limits<uint8_t>::max();
			case CV_16U: return std::numeric_limits<uint16_t>::max();

			case CV_8S: return std::numeric_limits<int8_t>::max();
			case CV_16S: return std::numeric_limits<int16_t>::max();
			case CV_32S: return std::numeric_limits<int32_t>::max();

			case CV_32F: return 1;
			case CV_64F: default: return 1;
			}
		}

		float getMaxVal(const cv::Mat& mat) {
			return sitara::opencv::getMaxVal(mat.depth());
		}

		cv::Vec3b convertColor(cv::Vec3b color, int code) {
			cv::Mat_<cv::Vec3b> mat(1, 1, CV_8UC3);
			mat(0, 0) = color;
			cv::cvtColor(mat, mat, code);
			return mat(0, 0);
		}

		ci::Color convertColor(ci::Color color, int code) {
			cv::Vec3b cvColor(color.r, color.g, color.b);
			cv::Vec3b result = sitara::opencv::convertColor(cvColor, code);
			return ci::Color(result[0], result[1], result[2]);
		}

		ci::ColorA convertColor(ci::ColorA color, int code) {
			cv::Vec3b cvColor(color.r, color.g, color.b);
			cv::Vec3b result = sitara::opencv::convertColor(cvColor, code);
			return ci::ColorA(result[0], result[1], result[2], color.a);
		}
	}
}