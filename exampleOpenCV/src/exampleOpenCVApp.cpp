#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "CinderOpenCV.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace ci;
using namespace ci::app;
using namespace std;

class exampleOpenCVApp : public App {
public:
	struct FaceId {
		cv::Rect mRect;
		long id;
		bool mMatched;
	};

	void setup() override;
	void mouseDown(MouseEvent event) override;
	void update() override;
	void draw() override;
	void detectFaces(cv::Mat& frame);

	static bool sortMaybeFaces(const cv::Rect& lhs, const cv::Rect& rhs) {
		// simple sorting function to speed up object tracking
		return lhs.x < rhs.x;
	}

	static bool sortFaces(const FaceId& lhs, const FaceId& rhs) {
		// simple sorting function to speed up object tracking
		return lhs.mRect.x < rhs.mRect.x;
	}


	gl::Texture2dRef mTexture;
	cv::VideoCapture mCamera;
	cv::CascadeClassifier mFaceClassifier;
	cv::CascadeClassifier mEyeClassifier;
	std::vector<cv::Rect> mClassifierOutput;
	std::vector<FaceId> mFaces;
	long mCurrentId;
	std::vector<cv::Rect> mEyes;
};

void exampleOpenCVApp::setup() {
	bool cameraFound = mCamera.open(0);
	std::printf("Camera Found: %s\n", cameraFound ? "true" : "false");
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	if (!mCamera.isOpened()) {
		std::printf("exampleOpenCVApp::setup Camera not found!\n");
	}

	mCurrentId = 0;
	mFaceClassifier.load("../assets/config/haarcascade_frontalface_alt.xml");
	mEyeClassifier.load("../assets/config/haarcascade_eye.xml");
}

void exampleOpenCVApp::mouseDown(MouseEvent event) {
}

void exampleOpenCVApp::update() {
	cv::Mat frame;
	mCamera >> frame;
	detectFaces(frame);

	cv::imwrite("../assets/images/output.png", frame);

	mTexture = gl::Texture::create(fromOcv(frame));
}

void exampleOpenCVApp::draw() {
	gl::clear();
	gl::draw(mTexture);
}

void exampleOpenCVApp::detectFaces(cv::Mat& frame) {
	cv::Mat frame_gray;
	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);  // Convert to gray scale
	cv::equalizeHist(frame_gray, frame);   	// Equalize histogram


	mFaceClassifier.detectMultiScale(frame_gray, mClassifierOutput, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	std::sort(mClassifierOutput.begin(), mClassifierOutput.end(), &sortMaybeFaces);
	std::sort(mFaces.begin(), mFaces.end(), &sortFaces);

	for (auto potentialFace : mClassifierOutput) {
		bool faceFound = false;
		for (auto face = mFaces.begin(); face != mFaces.end(); ++face) {
			if (abs(potentialFace.x - face->mRect.x) < 50 && abs(potentialFace.y - face->mRect.y) < 50) {
				faceFound = true;
				face->mMatched = true;
				face->mRect = potentialFace;
				break;
			}
		}

		if (!faceFound) {
			std::printf("New face %d found!\n", mCurrentId);
			FaceId newFace;
			newFace.mRect = potentialFace;
			newFace.id = mCurrentId;
			mCurrentId++;
			newFace.mMatched = true;
			mFaces.push_back(newFace);
		}
	}

	auto it = mFaces.begin();
	while(it != mFaces.end()) {
		if (!(it->mMatched)) {
			it = mFaces.erase(it);
		}
		else {
			it->mMatched = false;
			++it;
		}
	}

	// Iterate over all of the faces
	for (size_t i = 0; i < mFaces.size(); i++) {
		// Find center of faces
		cv::Point center(mFaces[i].mRect.x + mFaces[i].mRect.width / 2, mFaces[i].mRect.y + mFaces[i].mRect.height / 2);

		cv::Mat face = frame_gray(mFaces[i].mRect);
		cv::ellipse(frame, center, cv::Size(mFaces[i].mRect.width / 2, mFaces[i].mRect.height / 2), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);
		cv::putText(frame, std::to_string(mFaces[i].id), center + cv::Point(0, mFaces[i].mRect.height), CV_FONT_HERSHEY_PLAIN, 6, cv::Scalar(255));
	}
}

CINDER_APP( exampleOpenCVApp, RendererGl, [](App::Settings* settings) { settings->setConsoleWindowEnabled(); })
