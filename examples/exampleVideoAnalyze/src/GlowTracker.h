#pragma once

#include "OpenCV.h"

class Glow : public sitara::opencv::RectFollower {
protected:
	ci::Color color;
	ci::Rectf rect;
public:
	Glow() {};
	void setup(const cv::Rect& track);
	void update(const cv::Rect& track);
	void kill();
	void draw();
};