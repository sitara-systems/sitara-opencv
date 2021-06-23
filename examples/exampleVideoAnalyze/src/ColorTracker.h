#pragma once

#include "OpenCV.h"

class ColorTracker : public sitara::opencv::RectFollower {
protected:
	ci::Color color;
	ci::Rectf rect;
public:
	ColorTracker() {};
	void setup(const cv::Rect& track);
	void update(const cv::Rect& track);
	void kill();
	void draw();
};
