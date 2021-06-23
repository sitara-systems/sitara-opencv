#include "ColorTracker.h"
#include "cinder/app/App.h"
#include "cinder/Rand.h"

void ColorTracker::setup(const cv::Rect& track) {
	color.set(ci::CM_HSV, ci::vec3(ci::randFloat(), ci::randFloat(), ci::randFloat()));
}

void ColorTracker::update(const cv::Rect& track) {
	rect.set(track.tl().x, track.tl().y, track.br().x, track.br().y);
}

void ColorTracker::kill() {
}

void ColorTracker::draw() {
	{
		ci::gl::color(color);
		ci::gl::drawStrokedRect(rect);
	}
}