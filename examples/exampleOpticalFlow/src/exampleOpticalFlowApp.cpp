#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Capture.h"
#include "OpenCV.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class exampleOpticalFlowApp : public App {
  public:
	void setup() override;
	void update() override;
	void draw() override;

	ci::CaptureRef mCameraCapture;
	ci::gl::Texture2dRef mTexture;
	sitara::opencv::FlowFarneback mFlow;
	ci::vec2 p1;
	ci::Rectf rect;
};

void exampleOpticalFlowApp::setup() {
	try {
		mCameraCapture = ci::Capture::create(640, 480);
		mCameraCapture->start();
	}
	catch (...) {
		std::cout << "Could not initialize the capture" << endl;
	}
}

void exampleOpticalFlowApp::update() {
	if (mCameraCapture->isCapturing()) {
		if (mCameraCapture->checkNewFrame()) {
			auto pixels = mCameraCapture->getSurface();
			if (!mTexture) {
				// Capture images come back as top-down, and it's more efficient to keep them that way
				mTexture = ci::gl::Texture::create(*mCameraCapture->getSurface(), ci::gl::Texture::Format().loadTopDown());
			}
			else {
				mTexture->update(*mCameraCapture->getSurface());
			}
			mFlow.calcOpticalFlow(sitara::opencv::toOcv(*mCameraCapture->getSurface()));
		}
	}
}

void exampleOpticalFlowApp::draw() {
	gl::clear(Color(0, 0, 0));
	if (!mTexture) {
		return;
	}
	else {
		ci::gl::draw(mTexture);
		mFlow.draw();
	}
}

CINDER_APP( exampleOpticalFlowApp, RendererGl, [=](cinder::app::App::Settings* settings) {
	settings->setTitle("OpenCV Optical Flow Example");
	settings->setWindowSize(640, 480);
	settings->setConsoleWindowEnabled();
	settings->setHighDensityDisplayEnabled(false);
});
