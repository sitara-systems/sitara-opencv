#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "OpenCV.h"

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
};

void exampleVideoAnalyzeApp::setup() {
	mVideoCapture = std::make_shared<cv::VideoCapture>(ci::app::getAssetPath("demo.mp4").string());
	if (!mVideoCapture->isOpened()) {
		std::cout << "ERROR : could not open file; please check path" << std::endl;
	}
	ci::app::setFrameRate(24);
}

void exampleVideoAnalyzeApp::mouseDown( MouseEvent event )
{
}

void exampleVideoAnalyzeApp::update() {
	if (mVideoCapture->isOpened()) {
		cv::Mat frame;
		bool success = mVideoCapture->read(frame);
		if (!success) {
			std::cout << "Video file is complete." << std::endl;
		}

		mTexture = ci::gl::Texture2d::create(ci::fromOcv(frame));
	}
}

void exampleVideoAnalyzeApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) );
	gl::draw(mTexture);
}

CINDER_APP( exampleVideoAnalyzeApp, RendererGl, [](App::Settings* settings) { settings->setConsoleWindowEnabled(); })
