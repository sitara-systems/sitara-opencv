#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "OpenCV.h"
#include "GlowTracker.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class exampleVideoAnalyzeApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;

	std::shared_ptr<cv::VideoCapture> mVideoCapture;
	ci::gl::Texture2dRef mTexture;
	sitara::opencv::ContourFinder mContourFinder;
	sitara::opencv::RectTrackerFollower<Glow> mTracker;
};

void exampleVideoAnalyzeApp::setup() {
	mVideoCapture = std::make_shared<cv::VideoCapture>(ci::app::getAssetPath("videos/example.mov").string());
	if (!mVideoCapture->isOpened()) {
		std::cout << "ERROR : could not open file; please check path" << std::endl;
	}
	ci::app::setFrameRate(24);
	mTracker.setPersistence(15);
	mTracker.setMaximumDistance(32);
}

void exampleVideoAnalyzeApp::mouseDown( MouseEvent event ){
}

void exampleVideoAnalyzeApp::update() {
	if (mVideoCapture->isOpened()) {
		cv::Mat frame;
		bool success = mVideoCapture->read(frame);
		if (!success) {
			std::cout << "Video file is complete." << std::endl;
		}

		mContourFinder.findContours(frame);
		mTracker.track(mContourFinder.getBoundingRects());
		mTexture = ci::gl::Texture2d::create(sitara::opencv::fromOcv(frame));
	}
}

void exampleVideoAnalyzeApp::draw() {
	gl::clear( Color( 0, 0, 0 ) );
	gl::color(ci::Color(1, 1, 1));
	gl::draw(mTexture);
	for (auto& f : mTracker.getFollowers()) {
		f.draw();
	}
}

CINDER_APP( exampleVideoAnalyzeApp, RendererGl, [](App::Settings* settings) {
	settings->setConsoleWindowEnabled();
	settings->setWindowSize(ci::vec2(320, 240));
})
