#pragma once

#include "opencv2/opencv.hpp"
#include "cinder/Cinder.h"
#include "cinder/gl/gl.h"	
#include "cinder/ImageIo.h"

namespace sitara {
	namespace opencv {
		class ImageTargetCvMat : public ci::ImageTarget {
		public:
			static std::shared_ptr<ImageTargetCvMat> createRef(cv::Mat* mat) { return std::shared_ptr<ImageTargetCvMat>(new ImageTargetCvMat(mat)); }

			virtual bool hasAlpha() const { return mMat->channels() == 4; }
			virtual void* getRowPointer(int32_t row) { return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(mMat->data) + row * mMat->step); }

		protected:
			ImageTargetCvMat(cv::Mat* mat);

			cv::Mat* mMat;
		};

		class ImageSourceCvMat : public ci::ImageSource {
		public:
			ImageSourceCvMat(const cv::Mat& mat)
				: ImageSource()
			{
				mWidth = mat.cols;
				mHeight = mat.rows;
				if ((mat.channels() == 3) || (mat.channels() == 4)) {
					setColorModel(ImageIo::CM_RGB);
					if (mat.channels() == 4)
						setChannelOrder(ImageIo::BGRA);
					else
						setChannelOrder(ImageIo::BGR);
				}
				else if (mat.channels() == 1) {
					setColorModel(ImageIo::CM_GRAY);
					setChannelOrder(ImageIo::Y);
				}

				switch (mat.depth()) {
				case CV_8U: setDataType(ImageIo::UINT8); break;
				case CV_16U: setDataType(ImageIo::UINT16); break;
				case CV_32F: setDataType(ImageIo::FLOAT32); break;
				default:
					throw ci::ImageIoExceptionIllegalDataType();
				}

				mRowBytes = (int32_t)mat.step;
				mData = reinterpret_cast<const uint8_t*>(mat.data);
			}

			void load(ci::ImageTargetRef target) {
				// get a pointer to the ImageSource function appropriate for handling our data configuration
				ImageSource::RowFunc func = setupRowFunc(target);

				const uint8_t* data = mData;
				for (int32_t row = 0; row < mHeight; ++row) {
					((*this).*func)(target, row, data);
					data += mRowBytes;
				}
			}

			const uint8_t* mData;
			int32_t				mRowBytes;
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////////////
		// ImageTargetCvMat
		inline ImageTargetCvMat::ImageTargetCvMat(cv::Mat* mat)
			: ImageTarget(), mMat(mat)
		{
			switch (mat->depth()) {
			case CV_8U: setDataType(ImageIo::UINT8); break;
			case CV_16U: setDataType(ImageIo::UINT16); break;
			case CV_32F: setDataType(ImageIo::FLOAT32); break;
			default:
				throw ci::ImageIoExceptionIllegalDataType();
			}

			switch (mat->channels()) {
			case 1:
				setColorModel(ImageIo::CM_GRAY);
				setChannelOrder(ImageIo::Y);
				break;
			case 3:
				setColorModel(ImageIo::CM_RGB);
				setChannelOrder(ImageIo::BGR);
				break;
			case 4:
				setColorModel(ImageIo::CM_RGB);
				setChannelOrder(ImageIo::BGRA);
				break;
			default:
				throw ci::ImageIoExceptionIllegalColorModel();
				break;
			}
		}

		inline cv::Mat toOcv(ci::ImageSourceRef sourceRef, int type = -1)
		{
			if (type == -1) {
				int depth = CV_8U;
				if (sourceRef->getDataType() == ci::ImageIo::UINT16)
					depth = CV_16U;
				else if ((sourceRef->getDataType() == ci::ImageIo::FLOAT32) || (sourceRef->getDataType() == ci::ImageIo::FLOAT16))
					depth = CV_32F;
				int channels = ci::ImageIo::channelOrderNumChannels(sourceRef->getChannelOrder());
				type = CV_MAKETYPE(depth, channels);
			}

			cv::Mat result(sourceRef->getHeight(), sourceRef->getWidth(), type);
			ci::ImageTargetRef target = ImageTargetCvMat::createRef(&result);
			sourceRef->load(target);
			cv::flip(result, result, 0);
			return result;
		}

		int getTargetChannelsFromCode(int conversionCode);
		float getMaxVal(int cvDepth);
		float getMaxVal(const cv::Mat& mat);

		cv::Vec3b convertColor(cv::Vec3b color, int code);
		ci::Color convertColor(ci::Color color, int code);
		ci::ColorA convertColor(ci::ColorA color, int code);

		inline int getChannels(int cvImageType) {
			return CV_MAT_CN(cvImageType);
		}

		inline int getChannels(const cv::Mat& mat) {
			return mat.channels();
		}

		inline void copyGray(const cv::Mat& src, cv::Mat& dst) {
			int channels = getChannels(src);
			if (channels == 4) {
				cv::cvtColor(src, dst, cv::COLOR_RGBA2GRAY);
			}
			else if (channels == 3) {
				cv::cvtColor(src, dst, cv::COLOR_RGB2GRAY);
			}
			else if (channels == 1) {
				dst = src.clone();
			}
		}

		inline cv::Mat toOcvRef(ci::Channel8u& channel)
		{
			return cv::Mat(channel.getHeight(), channel.getWidth(), CV_MAKETYPE(CV_8U, 1), channel.getData(), channel.getRowBytes());
		}

		inline cv::Mat toOcvRef(ci::Channel16u& channel)
		{
			return cv::Mat(channel.getHeight(), channel.getWidth(), CV_MAKETYPE(CV_16U, 1), channel.getData(), channel.getRowBytes());
		}

		inline cv::Mat toOcvRef(ci::Channel32f& channel)
		{
			return cv::Mat(channel.getHeight(), channel.getWidth(), CV_MAKETYPE(CV_32F, 1), channel.getData(), channel.getRowBytes());
		}

		inline cv::Mat toOcvRef(ci::Surface8u& surface)
		{
			return cv::Mat(surface.getHeight(), surface.getWidth(), CV_MAKETYPE(CV_8U, surface.hasAlpha() ? 4 : 3), surface.getData(), surface.getRowBytes());
		}

		inline cv::Mat toOcvRef(ci::Surface16u& surface)
		{
			return cv::Mat(surface.getHeight(), surface.getWidth(), CV_MAKETYPE(CV_16U, surface.hasAlpha() ? 4 : 3), surface.getData(), surface.getRowBytes());
		}

		inline cv::Mat toOcvRef(ci::Surface32f& surface)
		{
			return cv::Mat(surface.getHeight(), surface.getWidth(), CV_MAKETYPE(CV_32F, surface.hasAlpha() ? 4 : 3), surface.getData(), surface.getRowBytes());
		}

		inline ci::ImageSourceRef fromOcv(cv::Mat& mat)
		{
			return ci::ImageSourceRef(new ImageSourceCvMat(mat));
		}

		inline ci::ImageSourceRef fromOcv(cv::UMat& umat)
		{
			return ci::ImageSourceRef(new ImageSourceCvMat(umat.getMat(cv::ACCESS_READ)));
		}

		inline cv::Scalar toOcv(const ci::Color& color)
		{
			return CV_RGB(color.r * 255, color.g * 255, color.b * 255);
		}

		inline ci::vec2 fromOcv(const cv::Point2f& point)
		{
			return ci::vec2(point.x, point.y);
		}

		inline ci::vec3 fromOcv(const cv::Point3f& point)
		{
			return ci::vec3(point.x, point.y, point.z);
		}


		inline cv::Point2f toOcv(const ci::vec2& point)
		{
			return cv::Point2f(point.x, point.y);
		}

		inline cv::Point3f toOcv(const ci::vec3& point)
		{
			return cv::Point3f(point.x, point.y, point.z);
		}

		inline ci::ivec2 fromOcv(const cv::Point& point)
		{
			return ci::ivec2(point.x, point.y);
		}

		inline cv::Point toOcv(const ci::ivec2& point)
		{
			return cv::Point(point.x, point.y);
		}

		inline cv::Rect toOcv(const ci::Rectf& r)
		{
			return cv::Rect(r.x1, r.y1, r.getWidth(), r.getHeight());
		}

		inline ci::Rectf fromOcv(const cv::Rect& r)
		{
			return ci::Rectf(r.x, r.y, r.x + r.width, r.y + r.height);
		}

		inline std::vector<cv::Point2f> toOcv(const ci::PolyLine2f& polyline) {
			std::vector<cv::Point2f> contour(polyline.size());
			std::vector<ci::vec2> polyPoints = polyline.getPoints();
			for (int i = 0; i < polyline.size(); i++) {
				contour[i] = toOcv(polyPoints[i]);
			}
			return contour;
		}

		// no matching std::vector<cv::Point2f> -> PolyLine function; prefer the one below

		inline std::vector<cv::Point2f> toOcv(const std::vector<ci::vec2>& points) {
			std::vector<cv::Point2f> out(points.size());
			for (int i = 0; i < points.size(); i++) {
				out[i] = toOcv(points[i]);
			}
			return out;
		}

		inline std::vector<ci::vec2> fromOcv(const std::vector<cv::Point2f>& points) {
			std::vector<ci::vec2> out(points.size());
			for (int i = 0; i < points.size(); i++) {
				out[i] = fromOcv(points[i]);
			}
			return out;
		}

		inline std::vector<cv::Point> toOcv(const std::vector<ci::ivec2>& points) {
			std::vector<cv::Point> out(points.size());
			for (int i = 0; i < points.size(); i++) {
				out[i] = toOcv(points[i]);
			}
			return out;
		}

		inline std::vector<ci::ivec2> fromOcv(const std::vector<cv::Point>& points) {
			std::vector<ci::ivec2> out(points.size());
			for (int i = 0; i < points.size(); i++) {
				out[i] = fromOcv(points[i]);
			}
			return out;
		}

		inline std::vector<cv::Point3f> toOcv(const std::vector<ci::vec3>& points) {
			std::vector<cv::Point3f> out(points.size());
			for (int i = 0; i < points.size(); i++) {
				out[i] = toOcv(points[i]);
			}
			return out;
		}

		inline std::vector<ci::vec3> fromOcv(const std::vector<cv::Point3f>& points) {
			std::vector<ci::vec3> out(points.size());
			for (int i = 0; i < points.size(); i++) {
				out[i] = fromOcv(points[i]);
			}
			return out;
		}
	}
}