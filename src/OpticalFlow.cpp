#include "OpticalFlow.h"

using namespace sitara::opencv;

OpticalFlow::OpticalFlow() : hasFlow(false) {
}

OpticalFlow::~OpticalFlow(){
}

void OpticalFlow::calcOpticalFlow(const cv::Mat& lastImage, const cv::Mat& currentImage) {
	if (lastImage.channels() == 1 && currentImage.channels() == 1) {
		calcFlow(lastImage, currentImage);
	}
	else {
		copyGray(lastImage, last);
		copyGray(currentImage, curr);
		calcFlow(last, curr);
	}
	hasFlow = true;
}

//you can add subsequent images this way without having to store
//the previous one yourself
void OpticalFlow::calcOpticalFlow(const cv::Mat& nextImage){
    copyGray(nextImage, curr);
	if(last.size == curr.size){
		calcFlow(last, curr);
		hasFlow = true;
	}
    swap(curr, last);
}

void OpticalFlow::draw(){
	if(hasFlow) {
		drawFlow(ci::Rectf(0, 0, getWidth(), getHeight()));
	}
}
void OpticalFlow::draw(float x, float y){
	if(hasFlow){
		drawFlow(ci::Rectf(x, y, getWidth(), getHeight()));
	}
}
void OpticalFlow::draw(float x, float y, float width, float height){
	if(hasFlow){
		drawFlow(ci::Rectf(x,y,width,height));
	}
}
void OpticalFlow::draw(ci::Rectf rect){
	if(hasFlow){
		drawFlow(rect);
	}
}
int OpticalFlow::getWidth()  {
    return curr.cols;
}
int OpticalFlow::getHeight() {
    return curr.rows;
}
void OpticalFlow::resetFlow() {
    last = cv::Mat();
    curr = cv::Mat();
    hasFlow = false;
}

FlowPyrLK::FlowPyrLK()
:windowSize(32)
,maxLevel(3)
,maxFeatures(200)
,qualityLevel(0.01)
,minDistance(4)
,pyramidLevels(10)
,calcFeaturesNextFrame(true)
{
}

FlowPyrLK::~FlowPyrLK(){
}

void FlowPyrLK::setWindowSize(int winsize){
	this->windowSize = winsize;
}
void FlowPyrLK::setMaxLevel(int maxLevel){
	this->maxLevel = maxLevel;
}
void FlowPyrLK::setMaxFeatures(int maxFeatures){
	this->maxFeatures = maxFeatures;
}
void FlowPyrLK::setQualityLevel(float qualityLevel){
	this->qualityLevel = qualityLevel;
}
void FlowPyrLK::setMinDistance(int minDistance){
	this->minDistance = minDistance;
}

void FlowPyrLK::calcFlow(cv::Mat prev, cv::Mat next){
	if(!nextPts.empty() || calcFeaturesNextFrame){
		if(calcFeaturesNextFrame){
			calcFeaturesToTrack(prevPts, next);
			if (prevPts.empty()) {
				nextPts.clear();
				return;
			}
			calcFeaturesNextFrame = false;
		}else{
            swap(prevPts, nextPts);
		}
		nextPts.clear();

#if CV_MAJOR_VERSION>=2 && (CV_MINOR_VERSION>4 || (CV_MINOR_VERSION==4 && CV_SUBMINOR_VERSION>=1))
		if (prevPyramid.empty()) {
			buildOpticalFlowPyramid(prev,prevPyramid,cv::Size(windowSize, windowSize),10);
		}
		buildOpticalFlowPyramid(next,pyramid,cv::Size(windowSize, windowSize),10);
		calcOpticalFlowPyrLK(prevPyramid,
                                pyramid,
                                prevPts,
                                nextPts,
                                status,
                                err,
                                cv::Size(windowSize, windowSize),
                                maxLevel);
		prevPyramid = pyramid;
		pyramid.clear();
#else
		calcOpticalFlowPyrLK(prev,
                                next,
                                prevPts,
                                nextPts,
                                status,
                                err,
                                cv::Size(windowSize, windowSize),
                                maxLevel);
#endif
		status.resize(nextPts.size(),0);
	}else{
		calcFeaturesToTrack(nextPts, next);
	}
}

void FlowPyrLK::calcFeaturesToTrack(std::vector<cv::Point2f> & features, cv::Mat next){
	goodFeaturesToTrack(
                        next,
                        features,
                        maxFeatures,
                        qualityLevel,
                        minDistance
                        );
}

void FlowPyrLK::resetFeaturesToTrack(){
	calcFeaturesNextFrame=true;
}

void FlowPyrLK::setFeaturesToTrack(const std::vector<ci::vec2> & features){
	nextPts.resize(features.size());
	for(std::size_t i=0;i<features.size();i++){
		nextPts[i]= toOcv(features[i]);
	}
	calcFeaturesNextFrame = false;
}

void FlowPyrLK::setFeaturesToTrack(const std::vector<cv::Point2f> & features){
	nextPts = features;
	calcFeaturesNextFrame = false;
}

std::vector<ci::vec2> FlowPyrLK::getFeatures(){
	return fromOcv(prevPts);
}

std::vector<ci::vec2> FlowPyrLK::getCurrent(){
	std::vector<ci::vec2> ret;
    for(std::size_t i = 0; i < nextPts.size(); i++) {
		if(status[i]){
            ret.push_back(fromOcv(nextPts[i]));
		}
	}
	return ret;
}

std::vector<ci::vec2> FlowPyrLK::getMotion(){
	std::vector<ci::vec2> ret;
	for(std::size_t i = 0; i < prevPts.size(); i++) {
		if(status[i]){
			ret.push_back(fromOcv(nextPts[i])-fromOcv(prevPts[i]));
		}
	}
	return ret;
}

void FlowPyrLK::drawFlow(ci::Rectf rect) {
	ci::vec2 offset(rect.getX1(),rect.getY1());
	ci::vec2 scale(rect.getWidth()/getWidth(),rect.getHeight()/getHeight());
	for(std::size_t i = 0; i < prevPts.size(); i++) {
		if(status[i]){
			ci::gl::drawLine(fromOcv(prevPts[i])*scale+offset, fromOcv(nextPts[i])*scale+offset);
		}
	}
}

void FlowPyrLK::resetFlow(){
    OpticalFlow::resetFlow();
    resetFeaturesToTrack();
    prevPts.clear();
}

FlowFarneback::FlowFarneback()
:pyramidScale(0.5)
,numLevels(4)
,windowSize(8)
,numIterations(2)
,polyN(7)
,polySigma(1.5)
,farnebackGaussian(false)
{
}

FlowFarneback::~FlowFarneback(){
}

void FlowFarneback::setPyramidScale(float scale){
	if(scale < 0.0 || scale >= 1.0){
		std::cout << "FlowFarneback::setPyramidScale" << "setting scale to a number outside of 0 - 1" << std::endl;
	}
	this->pyramidScale = scale;
}

void FlowFarneback::setNumLevels(int levels){
	this->numLevels = levels;
}

void FlowFarneback::setWindowSize(int winsize){
	this->windowSize = winsize;
}

void FlowFarneback::setNumIterations(int interations){
	this->numIterations = interations;
}

void FlowFarneback::setPolyN(int polyN){
	this->polyN = polyN;
}

void FlowFarneback::setPolySigma(float polySigma){
	this->polySigma = polySigma;
}

void FlowFarneback::setUseGaussian(bool gaussian){
	this->farnebackGaussian = gaussian;
}

void FlowFarneback::resetFlow(){
    OpticalFlow::resetFlow();
	flow.setTo(0);
}

void FlowFarneback::drawFlowMatrix(cv::Mat flowMatrix) {
	int stepSize = 4; //TODO: make class-level parameteric
	for (int y = 0; y < flowMatrix.rows; y += stepSize) {
		for (int x = 0; x < flowMatrix.cols; x += stepSize) {
			ci::vec2 cur = ci::vec2(x, y);
			const cv::Vec2f& vec = flowMatrix.at<cv::Vec2f>(y, x);
			ci::vec2 pos = ci::vec2(x + vec[0], y + vec[1]);
			ci::gl::drawLine(cur, pos);
		}
	}
}

void FlowFarneback::calcFlow(cv::Mat prev, cv::Mat next){
	int flags = 0;
	if(hasFlow){
		flags = cv::OPTFLOW_USE_INITIAL_FLOW;
	}
	if(farnebackGaussian){
		flags |= cv::OPTFLOW_FARNEBACK_GAUSSIAN;
	}

	cv::calcOpticalFlowFarneback(prev,
		next,
		flow,
		pyramidScale,
		numLevels,
		windowSize,
		numIterations,
		polyN,
		polySigma,
		flags);
}

cv::Mat& FlowFarneback::getFlow() {
    if(!hasFlow) {
        flow = cv::Mat::zeros(1, 1, CV_32FC2);
    }
    return flow;
}

ci::vec2 FlowFarneback::getFlowOffset(int x, int y){
	if(!hasFlow){
		return ci::vec2(0, 0);
	}
	else if (x > flow.cols || y > flow.rows || x < 0 || y < 0) {
		return ci::vec2(0, 0);
	}
	const cv::Vec2f& vec = flow.at<cv::Vec2f>(y, x);
	return ci::vec2(vec[0], vec[1]);
}

ci::vec2 FlowFarneback::getFlowPosition(int x, int y){
	if(!hasFlow){
		return ci::vec2(0, 0);
	}
	else if (x > flow.cols || y > flow.rows || x < 0 || y < 0) {
		return ci::vec2(0, 0);
	}
	const cv::Vec2f& vec = flow.at<cv::Vec2f>(y, x);
	return ci::vec2(x + vec[0], y + vec[1]);
}

ci::vec2 FlowFarneback::getTotalFlow(){
	return getTotalFlowInRegion(ci::Rectf(0,0,flow.cols, flow.rows));
}

ci::vec2 FlowFarneback::getAverageFlow(){
	return getAverageFlowInRegion(ci::Rectf(0,0,flow.cols,flow.rows));
}

ci::vec2 FlowFarneback::getAverageFlowInRegion(ci::Rectf rect){
	float area = rect.calcArea();

    if (area > 0)
    {
        return getTotalFlowInRegion(rect) / area;
    }
    else
    {
        return ci::vec2(0, 0);
    }
}

ci::vec2 FlowFarneback::getTotalFlowInRegion(ci::Rectf region){
	if(!hasFlow){
		return ci::vec2(0, 0);
	}

	const cv::Scalar& sc = cv::sum(flow(toOcv(region)));
	return ci::vec2(sc[0], sc[1]);
}

void FlowFarneback::drawFlow(ci::Rectf rect){
	if(!hasFlow){
		return;
	}

	ci::vec2 offset(rect.getX1(), rect.getY1());
	ci::vec2 scale(rect.getWidth()/flow.cols, rect.getHeight()/flow.rows);
	int stepSize = 4; //TODO: make class-level parameteric
	for(int y = 0; y < flow.rows; y += stepSize) {
		for(int x = 0; x < flow.cols; x += stepSize) {
			ci::vec2 cur = ci::vec2(x, y) * scale + offset;
			ci::gl::drawLine(cur, getFlowPosition(x, y) * scale + offset);
		}
	}
}
