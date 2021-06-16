#include "GlowTracker.h"
#include "cinder/app/App.h"
#include "cinder/Rand.h"

void Glow::setup(const cv::Rect& track) {
	color.set(ci::CM_HSV, ci::vec3(ci::randFloat(), ci::randFloat(), ci::randFloat()));
}

void Glow::update(const cv::Rect& track) {
	rect.set(track.tl().x, track.tl().y, track.br().x, track.br().y);
}

void Glow::kill() {
}

void Glow::draw() {
	{
		ci::gl::color(color);
		ci::gl::drawStrokedRect(rect);
	}
}