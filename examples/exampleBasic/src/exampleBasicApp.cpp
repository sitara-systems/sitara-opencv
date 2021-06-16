#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "OpenCV.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class exampleBasicApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
};

void exampleBasicApp::setup() {
	std::cout << cv::getBuildInformation() << std::endl;
}

void exampleBasicApp::mouseDown( MouseEvent event )
{
}

void exampleBasicApp::update()
{
}

void exampleBasicApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) ); 
}

CINDER_APP(exampleBasicApp, RendererGl, [=](cinder::app::App::Settings* settings) { settings->setConsoleWindowEnabled(); });
