#pragma once

#include "OpenCVHelpers.h"
#include "Tracker.h"
#include "cinder/Vector.h"
#include "cinder/PolyLine.h"

namespace sitara {
	namespace opencv {
		enum TrackingColorMode {
			TRACK_COLOR_RGB,
			TRACK_COLOR_HSV,
			TRACK_COLOR_H,
			TRACK_COLOR_HS
		};

		class ContourFinder {
		public:
			ContourFinder();

			template <class T>
			void findContours(T& img) {
				findContours(toOcv(img));
			}

			void findContours(cv::Mat img);
			const std::vector<std::vector<cv::Point> >& getContours() const;
			const std::vector<ci::PolyLine2f>& getPolylines() const;
			const std::vector<cv::Rect>& getBoundingRects() const;

			unsigned int size() const;
			std::vector<cv::Point>& getContour(unsigned int i);
			ci::PolyLine2f& getPolyline(unsigned int i);

			cv::Rect getBoundingRect(unsigned int i) const;
			cv::Point2f getCenter(unsigned int i) const;        // center of bounding box (most stable)
			cv::Point2f getCentroid(unsigned int i) const;      // center of mass (less stable)
			cv::Point2f getAverage(unsigned int i) const;       // average of contour vertices (least stable)
			cv::Vec2f getBalance(unsigned int i) const;         // difference between centroid and center
			double getContourArea(unsigned int i) const;
			double getArcLength(unsigned int i) const;
			std::vector<cv::Point> getConvexHull(unsigned int i) const;
			//std::vector<cv::Vec4i> getConvexityDefects(unsigned int i) const;
			cv::RotatedRect getMinAreaRect(unsigned int i) const;
			cv::Point2f getMinEnclosingCircle(unsigned int i, float& radius) const;
			cv::RotatedRect getFitEllipse(unsigned int i) const;
			std::vector<cv::Point> getFitQuad(unsigned int i) const;
			bool getHole(unsigned int i) const;
			cv::Vec2f getVelocity(unsigned int i) const;

			RectTracker& getTracker();
			unsigned int getLabel(unsigned int i) const;

			// Performs a point-in-contour test.
			// The function determines whether the point is inside a contour, outside, or lies on an edge (or coincides with a vertex)
			// The return value is the signed distance (positive stands for inside).
			double pointPolygonTest(unsigned int i, cv::Point2f point);

			void setThreshold(float thresholdValue);
			void setAutoThreshold(bool autoThreshold);
			void setInvert(bool invert);
			void setUseTargetColor(bool useTargetColor);
			void setTargetColor(ci::Color targetColor, TrackingColorMode trackingColorMode = TRACK_COLOR_RGB);
			void setFindHoles(bool findHoles);
			void setSortBySize(bool sortBySize);

			void resetMinArea();
			void resetMaxArea();
			void setMinArea(float minArea);
			void setMaxArea(float maxArea);
			void setMinAreaRadius(float minAreaRadius);
			void setMaxAreaRadius(float maxAreaRadius);
			void setMinAreaNorm(float minAreaNorm);
			void setMaxAreaNorm(float maxAreaNorm);

			void setSimplify(bool simplify);

			void draw() const;

		protected:
			cv::Mat hsvBuffer, thresh;
			bool autoThreshold, invert, simplify;
			float thresholdValue;

			bool useTargetColor;
			TrackingColorMode trackingColorMode;
			ci::Color targetColor;

			float minArea, maxArea;
			bool minAreaNorm, maxAreaNorm;

			std::vector<std::vector<cv::Point> > contours;
			std::vector<ci::PolyLine2f> polylines;

			RectTracker tracker;
			std::vector<cv::Rect> boundingRects;
			std::vector<bool> holes;

			int contourFindingMode;
			bool sortBySize;
		};
	}
}