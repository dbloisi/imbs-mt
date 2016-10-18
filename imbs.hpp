/*
*  IMBS-MT Background Subtraction Library multi-thread
*  Copyright 2016 Domenico Daniele Bloisi
*
*  This file is part of IMBS and it is distributed under the terms of the
*  GNU Lesser General Public License (Lesser GPL)
*
*
*
*  IMBS-MT is free software: you can redistribute it and/or modify
*  it under the terms of the GNU Lesser General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  IMBS-MT is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU Lesser General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public License
*  along with IMBS-MT.  If not, see <http://www.gnu.org/licenses/>.
*
*  This file contains the C++ OpenCV based implementation for
*  IMBS-MT algorithm described in
*
*  Domenico D. Bloisi, Andrea Pennisi, and Luca Iocchi
*  "Parallel Multi-modal Background Modeling"
*  Pattern Recognition Letters
*
*  Please, cite the above paper if you use IMBS-MT.
*
*  IMBS-MT has been written by Domenico D. Bloisi and Andrea Pennisi
*
*  Please, report suggestions/comments/bugs to
*  domenico.bloisi@gmail.com
*
*/

#ifndef __IMBS_HPP__
#define __IMBS_HPP__

//OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/features2d/features2d.hpp>

//C++
#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <functional>

using namespace cv;
using namespace std;

class BackgroundSubtractorIMBS
{
public:
    //! the default constructor
    BackgroundSubtractorIMBS();
    //! the full constructor
    BackgroundSubtractorIMBS(double fps,
			unsigned int fgThreshold=20,
			unsigned int associationThreshold=5,
			double samplingPeriod=500.,
			unsigned int minBinHeight=2,
			unsigned int numSamples=20,
			double alpha=0.65,
			double beta=1.15,
			double tau_s=60.,
			double tau_h=40.,
			double minArea=50.,
			double persistencePeriod=10000.,
			bool morphologicalFiltering=false
    		);
    //! the destructor
    ~BackgroundSubtractorIMBS();
    //! the update operator
    void apply(InputArray image, OutputArray fgmask, double learningRate=-1.);

    //! computes a background image which shows only the highest bin for each pixel
    void getBackgroundImage(OutputArray backgroundImage) const;

    //! re-initiaization method
    void initialize(Size frameSize, int frameType);
	
	bool loadBg(const char* filename);
	void saveBg(string* filename);

private:
    //method for creating the background model
    void createBg(unsigned int bg_sample_number);
    //method for creating the incremental background model
    void createIncrementalBg(unsigned int bg_sample_number);
    //method for updating the background model
    void updateBg();
    //method for computing the foreground mask
    void getFg();
    //method for computing the incremental foreground mask
    void getIncrementalFg();
    //method for refining foreground mask
    void filterFg();
    //method for filtering out blobs smaller than a given area
    void areaThresholding();
    //method for getting the current time
    double getTimestamp();
    //method for converting from RGB to HSV
    void convertImageRGBtoHSV(const Mat& imageRGB);
    //Utils//
    //abs function
    int abs_(int _value);
    //max between three numbers
    int max_(int _a, int _b, int _c);
    //min between two numbers
    int min_(int _a, int _b);
	
	

    //current input RGB frame
    Mat frame;
    vector<Mat> frameBGR;
    //frame size
    Size frameSize;
    //frame type
    int frameType;
    //total number of pixels in frame
    unsigned int numPixels;
    //current background sample
    Mat bgSample;
    vector<Mat> bgSampleBGR;
    //current background image which shows only the highest bin for each pixel
    //(just for displaying purposes)
    Mat bgImage;
    //current foreground mask
    Mat fgmask;
    Mat fgIncrementalMask;
    Mat fgMask_;

    Mat fgfiltered;
	
	string* bgFilename;
	bool loadedBg;
	
    //number of fps
    double fps;
    //time stamp in milliseconds (ms)
    double timestamp;
    //previous time stamp in milliseconds (ms)
    double prev_timestamp;
    double initial_tick_count;
	  //initial message to be shown until the first bg model is ready 
    Mat initialMsgGray;
    Mat initialMsgRGB;
    Mat imageHSV;

	float FLOAT_TO_BYTE;
	float BYTE_TO_FLOAT;
	float fhV1;
	float fhV2;
	
    //struct for modeling the background values for a single pixel
    typedef struct {
      vector<Vec3b> binValues;
      vector<uchar> binHeights;
      vector<bool> isFg;
    } Bins;
	  
    vector<Bins> bgBins;
    vector<Bins> incrementalBgBins;
public:
    //struct for modeling the background values for the entire frame
	typedef struct {
          vector<Vec3b> values;
          vector<bool> isValid;
          vector<bool> isFg;
          vector<uchar> counter;
	} BgModel;
	
	//bool isBackgroundCreated;
private:
    vector<BgModel> bgModel;
    vector<BgModel> incrementalBgModel;
    //method for suppressing shadows and highlights
    void hsvSuppression(const vector<BgModel> &_bgModel);

	//SHADOW SUPPRESSION PARAMETERS
	float alpha;
	float beta;
	uchar tau_s;
	uchar tau_h;

	unsigned int minBinHeight;
	unsigned int numSamples;
    unsigned int incrementalNumSamples;
	unsigned int samplingPeriod;
    unsigned int incrementalSamplingPeriod;
    unsigned int increamental_reduction_variable;
	unsigned long prev_bg_frame_time;
    unsigned long prev_incremental_bg_frame_time;
	unsigned int bg_frame_counter;
    unsigned int incremental_bg_frame_counter;
	unsigned int associationThreshold;
	unsigned int maxBgBins;
	unsigned int nframes;

	double minArea;
	bool bg_reset;
	unsigned int persistencePeriod;
    bool prev_area;
    bool incremental_bg;
	unsigned int fgThreshold;
	uchar SHADOW_LABEL;
	uchar PERSISTENCE_LABEL;
	uchar FOREGROUND_LABEL;
    //persistence map
    vector<unsigned int> persistenceMap;
    Mat persistenceImage;

    vector<unsigned int> counters;
    vector<unsigned int> incrementalCounters;
    vector<unsigned int> counterCopy;

    bool morphologicalFiltering;

public:
    unsigned int getMaxBgBins() {
    	return maxBgBins;
    }
    unsigned int getFgThreshold() {
        return fgThreshold;
    }
    void getBgModel(vector<BgModel> &bgModel_copy);
};

#endif //__IMBS_HPP__
