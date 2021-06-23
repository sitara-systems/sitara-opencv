#pragma once

#include "OpenCVHelpers.h"

namespace sitara {
	namespace opencv {
		class OpticalFlow {
		public:
			// should constructor be protected?
			OpticalFlow();
			virtual ~OpticalFlow();

			//call these functions to calculate flow on sequential images.
			//After this call the flow field will be populated and
			//subsequent calls to getFlow() will be updated
			void calcOpticalFlow(const cv::Mat& lastImage, const cv::Mat& currentImage);

			//call with subsequent images to do running optical flow.
			void calcOpticalFlow(const cv::Mat& nextImage);

			void draw();
			void draw(float x, float y);
			void draw(float x, float y, float width, float height);
			void draw(ci::Rectf rect);
			int getWidth();
			int getHeight();

			virtual void resetFlow();

		private:
			cv::Mat last;
			cv::Mat curr;

		protected:
			bool hasFlow;

			//specific flow implementation
			virtual void calcFlow(cv::Mat prev, cv::Mat next) = 0;
			//specific drawing implementation
			virtual void drawFlow(ci::Rectf r) = 0;
		};

		//there are two implementations of Flow
		//use Farneback for a dense flow field,
		//use PyrLK for specific features

		//see http://opencv.willowgarage.com/documentation/cpp/motion_analysis_and_object_tracking.html
		//for more info on the meaning of these parameters

		class FlowPyrLK : public OpticalFlow {
		public:
			FlowPyrLK();
			virtual ~FlowPyrLK();

			//flow parameters
			void setMinDistance(int minDistance);
			void setWindowSize(int winsize);

			//feature finding parameters
			void setMaxLevel(int maxLevel);
			void setMaxFeatures(int maxFeatures);
			void setQualityLevel(float qualityLevel);
			void setPyramidLevels(int levels);

			//returns tracking features for this image
			std::vector<ci::vec2> getFeatures();
			std::vector<ci::vec2> getCurrent();
			std::vector<ci::vec2> getMotion();

			// recalculates features to track
			void resetFeaturesToTrack();
			void setFeaturesToTrack(const std::vector<ci::vec2>& features);
			void setFeaturesToTrack(const std::vector<cv::Point2f>& features);
			void resetFlow();
		protected:

			void drawFlow(ci::Rectf r);
			void calcFlow(cv::Mat prev, cv::Mat next);
			void calcFeaturesToTrack(std::vector<cv::Point2f>& features, cv::Mat next);

			std::vector<cv::Point2f> prevPts, nextPts;

			//LK feature finding parameters
			int windowSize;
			int maxLevel;
			int maxFeatures;
			float qualityLevel;

			//min distance for PyrLK
			int minDistance;

			//pyramid levels
			int pyramidLevels;

			bool calcFeaturesNextFrame;

			//pyramid + err/status data
			std::vector<cv::Mat> pyramid;
			std::vector<cv::Mat> prevPyramid;
			std::vector<uchar> status;
			std::vector<float> err;
		};

		class FlowFarneback : public OpticalFlow {
		public:

			FlowFarneback();
			virtual ~FlowFarneback();

			//see http://opencv.willowgarage.com/documentation/cpp/motion_analysis_and_object_tracking.html
			//for a description of these parameters
			void setPyramidScale(float scale);
			void setNumLevels(int levels);
			void setWindowSize(int winsize);
			void setNumIterations(int interations);
			void setPolyN(int polyN);
			void setPolySigma(float polySigma);
			void setUseGaussian(bool gaussian);

			cv::Mat& getFlow();
			ci::vec2 getTotalFlow();
			ci::vec2 getAverageFlow();
			ci::vec2 getFlowOffset(int x, int y);
			ci::vec2 getFlowPosition(int x, int y);
			ci::vec2 getTotalFlowInRegion(ci::Rectf region);
			ci::vec2 getAverageFlowInRegion(ci::Rectf region);

			//call this if you switch to a new video file to reset internal caches
			void resetFlow();
			void drawFlowMatrix(cv::Mat flowMatrix);
		protected:
			cv::Mat flow;

			void drawFlow(ci::Rectf rect);
			void calcFlow(cv::Mat prev, cv::Mat next);

			float pyramidScale;
			int numLevels;
			int windowSize;
			int numIterations;
			int polyN;
			float polySigma;
			bool farnebackGaussian;
		};
	}
}
