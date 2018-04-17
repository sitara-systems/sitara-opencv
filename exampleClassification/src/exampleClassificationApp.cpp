#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Rand.h"

#include <opencv2\core\core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\ml\ml.hpp>

using namespace ci;
using namespace ci::app;
using namespace std;

class exampleClassificationApp : public App {
  public:
	void setup() override;
	void update() override;
	void draw() override;

	std::vector<cv::Point> mPoints;
	std::vector<int> mLabels;
	cv::Ptr<cv::ml::SVM> mSVM;
};

void exampleClassificationApp::setup() {
	for (int i = 0; i < 100; i++) {
		float x = Rand::randFloat(0, ci::app::getWindowWidth());
		float y = Rand::randFloat(0, x);
		mPoints.push_back(cv::Point(x, y));
		mLabels.push_back(-1.0);
	}
	for (int i = 0; i < 100; i++) {
		float x = Rand::randFloat(0, ci::app::getWindowWidth());
		float y = Rand::randFloat(x, ci::app::getWindowHeight());
		mPoints.push_back(cv::Point(x, y));
		mLabels.push_back(1.0);
	}
	cv::Mat trainingData(mPoints.size(), 2, CV_32FC1, mPoints.data());
	cv::Mat trainingLabels(mLabels.size(), 1, CV_32SC1, mLabels.data());

	std::printf("Created SVM\n");
	mSVM = cv::ml::SVM::create();

	std::printf("Setting parameters...\n");
	mSVM->setType(cv::ml::SVM::C_SVC);
	mSVM->setKernel(cv::ml::SVM::LINEAR);
	mSVM->setC(1.0);

	std::printf("Training...\n");
	mSVM->train(trainingData, cv::ml::ROW_SAMPLE, trainingLabels);

	float data[2] = { 0, ci::app::getWindowWidth() / 2.0 };
	cv::Mat query(1, 2, CV_32F, data);
	int response = mSVM->predict(query);

	std::printf("Asked SVM where to classify %f, %f -- result was %d\n", query.at<float>(0,0), query.at<float>(0, 1), response);
}

void exampleClassificationApp::update() {

}

void exampleClassificationApp::draw() {
	gl::clear( Color( 0, 0, 0 ) );
	for (int i = 0; i < mPoints.size(); i++) {
		if (mLabels[i] == -1.0) {
			gl::color(Color(1, 0, 0));
		}
		else {
			gl::color(Color(0, 0, 1));
		}
		ci::gl::drawSolidCircle(ci::vec2(mPoints[i].x, mPoints[i].y), 5);
	}
	cv::Mat supports;
	if (mSVM->isTrained()) {
		supports = mSVM->getSupportVectors();
		std::cout << "Suuport Vectors are " << supports << std::endl;
	}

}

CINDER_APP( exampleClassificationApp, RendererGl, [](App::Settings* settings) { settings->setConsoleWindowEnabled(); })
