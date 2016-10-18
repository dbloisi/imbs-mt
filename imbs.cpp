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

#include "imbs.hpp"

using namespace std;
using namespace cv;

BackgroundSubtractorIMBS::BackgroundSubtractorIMBS()
{
    fps = 25.;
    fgThreshold = 15;
    associationThreshold = 5;
	samplingPeriod = 400.;//ms
    minBinHeight = 2;
	numSamples = 30;
    alpha = 0.65;
    beta = 1.15;
    tau_s = 60.;
    tau_h = 40.;
    minArea = 50.;
	persistencePeriod = samplingPeriod * numSamples / 3.0;//ms

	initial_tick_count = (double)getTickCount();

	//morphological Opening and closing
	morphologicalFiltering = false;
	
	bgFilename = NULL;
	loadedBg = false;
}

BackgroundSubtractorIMBS::BackgroundSubtractorIMBS(
		double fps,
		unsigned int fgThreshold,
		unsigned int associationThreshold,
		double samplingPeriod,
		unsigned int minBinHeight,
		unsigned int numSamples,
		double alpha,
		double beta,
		double tau_s,
		double tau_h,
		double minArea,
		double persistencePeriod,
		bool morphologicalFiltering)
{
	this->fps = fps;
	this->fgThreshold = fgThreshold;
	this->persistencePeriod = persistencePeriod;
	if(minBinHeight <= 1){
		this->minBinHeight = 1;
	}
	else {
		this->minBinHeight = minBinHeight;
	}
	this->associationThreshold = associationThreshold;
	this->samplingPeriod = samplingPeriod;//ms
	this->minBinHeight = minBinHeight;
	this->numSamples = numSamples;
	this->alpha = alpha;
	this->beta = beta;
	this->tau_s = tau_s;
	this->tau_h = tau_h;
	this->minArea = minArea;

    increamental_reduction_variable = 10;

    incrementalSamplingPeriod = 100;
    if(incrementalSamplingPeriod > samplingPeriod)
    {
		incrementalSamplingPeriod = samplingPeriod;
    }

    incrementalNumSamples = numSamples / increamental_reduction_variable;

    if(incrementalNumSamples < 6)
    {
        incrementalNumSamples = 6;
    }


	if(fps == 0.)
		initial_tick_count = (double)getTickCount();
	else
		initial_tick_count = 0;

	//morphological Opening and closing
	this->morphologicalFiltering = morphologicalFiltering;
	
	bgFilename = NULL;
	loadedBg = false;
}

BackgroundSubtractorIMBS::~BackgroundSubtractorIMBS()
{
}

void BackgroundSubtractorIMBS::initialize(Size frameSize, int frameType)
{
	if(loadedBg)
		return;
	cout << "INPUT: WIDTH " << frameSize.width << "  HEIGHT " << frameSize.height <<
			"  FPS " << fps << endl;
	cout << endl;

	this->frameSize = frameSize;
	this->frameType = frameType;
    this->numPixels = frameSize.area();

    persistenceMap.resize(numPixels, 0);
    counters.resize(numPixels, 0);
    incrementalCounters.resize(numPixels, 0);

    bgBins.resize(numPixels);
    bgModel.resize(numPixels);

    incrementalBgBins.resize(numPixels);
    incrementalBgModel.resize(numPixels);

	maxBgBins = numSamples / minBinHeight;

	timestamp = 0.;//ms
	prev_timestamp = 0.;//ms
	prev_bg_frame_time = 0;
    prev_incremental_bg_frame_time = 0;
	bg_frame_counter = 0;
    incremental_bg_frame_counter = 0;
    prev_area = 0;
    incremental_bg = true;

	SHADOW_LABEL = 80;
	PERSISTENCE_LABEL = 180;
	FOREGROUND_LABEL = 255;

	FLOAT_TO_BYTE = 255.0f;
	BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
	fhV1 = 2.0f / 6.0f;
	fhV2 = 4.0f / 6.0f;

	fgmask.create(frameSize, CV_8UC1);
	fgfiltered.create(frameSize, CV_8UC1);
	persistenceImage = Mat::zeros(frameSize, CV_8UC1);
	bgSample.create(frameSize, CV_8UC3);
	bgImage = Mat::zeros(frameSize, CV_8UC3);

	//initial message to be shown until the first fg mask is computed
	initialMsgGray = Mat::zeros(frameSize, CV_8UC1);
	putText(initialMsgGray, "Creating", Point(10,20), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgGray, "initial", Point(10,40), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgGray, "background...", Point(10,60), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	
	initialMsgRGB = Mat::zeros(frameSize, CV_8UC3);
	putText(initialMsgRGB, "Creating", Point(10,20), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgRGB, "initial", Point(10,40), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));
	putText(initialMsgRGB, "background...", Point(10,60), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255, 255, 255));

	if(minBinHeight <= 1){
		minBinHeight = 1;
	}

	for(unsigned int p = 0; p < numPixels; ++p)
	{
        bgBins[p].binValues.resize(numSamples);
        bgBins[p].binHeights.resize(numSamples);
        bgBins[p].isFg.resize(numSamples, false);
		
        bgModel[p].values.resize(maxBgBins);
        bgModel[p].isValid.resize(maxBgBins);
		bgModel[p].isValid[0] = false;
        bgModel[p].isFg.resize(maxBgBins, false);
        bgModel[p].counter.resize(maxBgBins);

        incrementalBgBins[p].binValues.resize(numSamples);
        incrementalBgBins[p].binHeights.resize(numSamples);
        incrementalBgBins[p].isFg.resize(numSamples, false);

        incrementalBgModel[p].values.resize(maxBgBins);
        incrementalBgModel[p].isValid.resize(maxBgBins);
        incrementalBgModel[p].isValid[0] = false;
        incrementalBgModel[p].isFg.resize(maxBgBins, false);
        incrementalBgModel[p].counter.resize(maxBgBins);
	}
}

void BackgroundSubtractorIMBS::apply(InputArray _frame, OutputArray _fgmask, double learningRate)
{
    frame = _frame.getMat();

	CV_Assert(frame.depth() == CV_8U);
    CV_Assert(frame.channels() == 3);

    bool needToInitialize = nframes == 0 || frame.type() != frameType;
    if( needToInitialize ) {
    	initialize(frame.size(), frame.type());
    }

    _fgmask.create(frameSize, CV_8UC1);
    fgmask = _fgmask.getMat();
    fgmask = Scalar(0);

    //get current time
    prev_timestamp = timestamp;
    if(fps == 0.) {
    	timestamp = getTimestamp();//ms
    }
    else {
    	timestamp += 1000./fps;//ms
    }

    //wait for the first model to be generated
    if(bgModel[0].isValid[0])
    {
        incremental_bg = false;
        std::thread hsvConversion(&BackgroundSubtractorIMBS::convertImageRGBtoHSV, this, frame);
        getFg();
        hsvConversion.join();
        hsvSuppression(bgModel);
        filterFg();
    }	

    if(incremental_bg && incrementalBgModel[0].isValid[0])
    {
        std::thread hsvConversion(&BackgroundSubtractorIMBS::convertImageRGBtoHSV, this, frame);
        getIncrementalFg();
        hsvConversion.join();
        hsvSuppression(incrementalBgModel);
        filterFg();
    }

	//update the bg model
    updateBg();
	
	//show an initial message if the first bg is not yet ready
    if(incremental_bg && !incrementalBgModel[0].isValid[0]) {
        initialMsgGray =
        initialMsgRGB = cv::Mat::zeros(frame.size(), CV_8UC1);
	}


    ++nframes;
}

void BackgroundSubtractorIMBS::updateBg() {

	if(prev_bg_frame_time > timestamp) {
		prev_bg_frame_time = timestamp;
	}

	if(bg_frame_counter == numSamples - 1) {
        if(incremental_bg)
        {
            bgImage = cv::Scalar(0, 0, 0);
        }
        createBg(bg_frame_counter);
		bg_frame_counter = 0;
	}
	else { //bg_frame_counter < (numSamples - 1)

		if((timestamp - prev_bg_frame_time) >= samplingPeriod)
		{
			//get a new sample for creating the bg model
			prev_bg_frame_time = timestamp;
			frame.copyTo(bgSample);
			createBg(bg_frame_counter);
			bg_frame_counter++;
		}
	}


    if(incremental_bg)
    {
        if(prev_incremental_bg_frame_time > timestamp) {
            prev_incremental_bg_frame_time = timestamp;
        }

        if(incremental_bg_frame_counter == incrementalNumSamples - 1) {
            createIncrementalBg(incremental_bg_frame_counter);
            incremental_bg_frame_counter = 0;
        }
        else { //incremental_bg_frame_counter < (numSamples - 1)

            if((timestamp - prev_incremental_bg_frame_time) >= incrementalSamplingPeriod)
            {
                //get a new sample for creating the bg model
                prev_incremental_bg_frame_time = timestamp;
                frame.copyTo(bgSample);
                createIncrementalBg(incremental_bg_frame_counter);
                incremental_bg_frame_counter++;
            }
        }
    }


}

double BackgroundSubtractorIMBS::getTimestamp() {
	return ((double)getTickCount() - initial_tick_count)*1000./getTickFrequency();
}

void BackgroundSubtractorIMBS::hsvSuppression(const vector<BgModel> &_bgModel) {

	uchar h_i, s_i, v_i;
	uchar h_b, s_b, v_b;
	float h_diff, s_diff, v_ratio;

	int bB, bG, bR;
	float fDelta;
	float fMin, fMax;
	int iMax;

	float fR;
	float fG;
	float fB;
	float fH;
	float fS;
	float fV;

	
	vector<Mat> imHSV;
    cv::split(imageHSV, imHSV);

	for(unsigned int p = 0; p < numPixels; ++p) {
		if(fgmask.data[p]) {
			//get the hsv values for the input image
			h_i = imHSV[0].data[p];
			s_i = imHSV[1].data[p];
			v_i = imHSV[2].data[p];
			
            for(unsigned int n = 0; n < counterCopy[p]; ++n) {
                if(!_bgModel[p].isValid[n]) {
					break;
				}
                if(_bgModel[p].isFg[n]) {
					continue;
				}
				//get the hsv values for the background mode

				// Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order
				bB = _bgModel[p].values[n].val[0];
				bG = _bgModel[p].values[n].val[1];
				bR = _bgModel[p].values[n].val[2];
				// Convert from 8-bit integers to floats.
				fR = bR * BYTE_TO_FLOAT;
				fG = bG * BYTE_TO_FLOAT;
				fB = bB * BYTE_TO_FLOAT;

				// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
				
				// Get the min and max, but use integer comparisons for slight speedup.
				if (bB < bG) {
					if (bB < bR) {
						fMin = fB;
						if (bR > bG) {
							iMax = bR;
							fMax = fR;
						}
						else {
							iMax = bG;
							fMax = fG;
						}
					}
					else {
						fMin = fR;
						fMax = fG;
						iMax = bG;
					}
				}
				else {
					if (bG < bR) {
						fMin = fG;
						if (bB > bR) {
							fMax = fB;
							iMax = bB;
						}
						else {
							fMax = fR;
							iMax = bR;
						}
					}
					else {
						fMin = fR;
						fMax = fB;
						iMax = bB;
					}
				}
				fDelta = fMax - fMin;
				fV = fMax;				// Value (Brightness).
				if (iMax != 0) {			// Make sure its not pure black.
					fS = fDelta / fMax;		// Saturation.
					float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta);	// Make the Hues between 0.0 to 1.0 instead of 6.0
					if (iMax == bR) {		// between yellow and magenta.
						fH = (fG - fB) * ANGLE_TO_UNIT;
					}
					else if (iMax == bG) {		// between cyan and yellow.
						fH = fhV1 + (fB - fR) * ANGLE_TO_UNIT;
					}
					else {				// between magenta and cyan.
						fH = fhV2 + (fR - fG) * ANGLE_TO_UNIT;
					}
					// Wrap outlier Hues around the circle.
					if (fH < 0.0f)
						fH += 1.0f;
					if (fH >= 1.0f)
						fH -= 1.0f;
				}
				else {
					// color is pure Black.
					fS = 0;
					fH = 0;	// undefined hue
				}

				// Convert from floats to 8-bit integers.
				int bH = (int)(0.5f + fH * 255.0f);
				int bS = (int)(0.5f + fS * 255.0f);
				int bV = (int)(0.5f + fV * 255.0f);

				// Clip the values to make sure it fits within the 8bits.
				if (bH > 255)
					bH = 255;
				if (bH < 0)
					bH = 0;
				if (bS > 255)
					bS = 255;
				if (bS < 0)
					bS = 0;
				if (bV > 255)
					bV = 255;
				if (bV < 0)
					bV = 0;
				
				h_b = bH;
				s_b = bS;
				v_b = bV;

				v_ratio = (float)v_i / (float)v_b;
                s_diff = this->abs_(s_i - s_b);
                int a = this->abs_(h_i - h_b);
                int b = 255 - this->abs_(h_i - h_b);
                h_diff = this->min_(a, b);

				if(	h_diff <= tau_h &&
					s_diff <= tau_s &&
					v_ratio >= alpha &&
					v_ratio < beta)
				{
					fgmask.data[p] = SHADOW_LABEL;
					break;
				}
			}//for
		}//if
	}//numPixels
}

void BackgroundSubtractorIMBS::createBg(unsigned int bg_sample_number) {
    if(!bgSample.data) {
	//cerr << "createBg -- an error occurred: " <<
	//		" unable to retrieve frame no. " << bg_sample_number << endl;

	//TODO vedere gestione errori
	abort();
    }
    Vec3b currentPixel;
    //split bgSample in channels
    cv::split(bgSample, bgSampleBGR);
	
    //create a statistical model for each pixel (a set of bins of variable size)
    for(unsigned int p = 0; p < numPixels; ++p) {
	//create an initial bin for each pixel from the first sample (bg_sample_number = 0)
	if(bg_sample_number == 0) {

            bgBins[p].binValues[0][0] = bgSampleBGR[0].data[p];
            bgBins[p].binValues[0][1] = bgSampleBGR[1].data[p];
            bgBins[p].binValues[0][2] = bgSampleBGR[2].data[p];

            bgBins[p].binHeights[0] = 1;
            std::fill(bgBins[p].binHeights.begin() + 1, bgBins[p].binHeights.end(), 0);

			//if the sample pixel is from foreground keep track of that situation
            bgBins[p].isFg[0] = (!incremental_bg && fgmask.data[p] == FOREGROUND_LABEL) ? true : false;

	}//if(bg_sample_number == 0)
        //if all samples have been processed
	//it is time to compute the fg mask
	else if(bg_sample_number == (numSamples - 1)) {
	    unsigned int index = 0;
	    int max_height = -1;

	    for(unsigned int s = 0; s < numSamples; ++s){
                if(bgBins[p].binHeights[s] == 0) {
		    bgModel[p].isValid[index] = false;
		    break;
		}
		if(index == maxBgBins) {
                    --counters[p];
		    break;
		}
		else if(bgBins[p].binHeights[s] >= minBinHeight) {
		    if(bgBins[p].binHeights[s] > max_height) {
                        max_height = bgBins[p].binHeights[s];

                        bgModel[p].values[index][0] = bgModel[p].values[0][0];
                        bgModel[p].values[index][1] = bgModel[p].values[0][1];
                        bgModel[p].values[index][2] = bgModel[p].values[0][2];

			bgModel[p].isValid[index] = true;
			bgModel[p].isFg[index] = bgModel[p].isFg[0];
			bgModel[p].counter[index] = bgModel[p].counter[0];

                        bgModel[p].values[0][0] = bgBins[p].binValues[s][0];
                        bgModel[p].values[0][1] = bgBins[p].binValues[s][1];
                        bgModel[p].values[0][2] = bgBins[p].binValues[s][2];

			bgModel[p].isValid[0] = true;
			bgModel[p].isFg[0] = bgBins[p].isFg[s];
			bgModel[p].counter[0] = bgBins[p].binHeights[s];
		    }
		    else {
                        bgModel[p].values[index][0] = bgBins[p].binValues[s][0];
                        bgModel[p].values[index][1] = bgBins[p].binValues[s][1];
                        bgModel[p].values[index][2] = bgBins[p].binValues[s][2];

			bgModel[p].isValid[index] = true;
			bgModel[p].isFg[index] = bgBins[p].isFg[s];
			bgModel[p].counter[index] = bgBins[p].binHeights[s];
		    }
                    ++counters[p];
		    ++index;
		}
	    } //for all numSamples
	}//bg_sample_number == (numSamples - 1)
	else { //bg_sample_number > 0 && bg_sample_number != (numSamples - 1)
            currentPixel[0] = bgSampleBGR[0].data[p];
            currentPixel[1] = bgSampleBGR[1].data[p];
            currentPixel[2] = bgSampleBGR[2].data[p];

	    int den = 0;
	    for(unsigned int s = 0; s < bg_sample_number; ++s) {
		//try to associate the current pixel values to an existing bin
                if( this->abs_(currentPixel[2] - bgBins[p].binValues[s][2]) <= (int) associationThreshold &&
                    this->abs_(currentPixel[1] - bgBins[p].binValues[s][1]) <= (int) associationThreshold &&
                    this->abs_(currentPixel[0] - bgBins[p].binValues[s][0]) <= (int) associationThreshold )
				{
                    den = (bgBins[p].binHeights[s] + 1);
		    for(int k = 0; k < 3; ++k) {
			bgBins[p].binValues[s][k] =
                            (bgBins[p].binValues[s][k] * bgBins[p].binHeights[s] + currentPixel[k]) / den;
		    }
		    bgBins[p].binHeights[s]++; //increment the height of the bin
                    if(!incremental_bg && fgmask.data[p] == FOREGROUND_LABEL) {
                        bgBins[p].isFg[s] = true;
		    }
		    break;
		}
		//if the association is not possible, create a new bin
		else if(bgBins[p].binHeights[s] == 0)	{

                    bgBins[p].binValues[s] = currentPixel;
		    bgBins[p].binHeights[s]++;
                    bgBins[p].isFg[s] = (!incremental_bg && fgmask.data[p] == FOREGROUND_LABEL) ? true : false;
                    break;
		}
	    }//for(unsigned int s = 0; s <= bg_sample_number; ++s)
	}//else --> bg_sample_number > 0 && bg_sample_number != (numSamples - 1)
    }//numPixels

    if(bg_sample_number == (numSamples - 1)) {
        std::cout << "new STABLE bg created" << std::endl;
	persistenceImage = Scalar(0);
		
        counterCopy = counters;
        std::fill(persistenceMap.begin(), persistenceMap.end(), 0);
        std::fill(counters.begin(), counters.end(), 0);

	unsigned int p = 0;
	for(int i = 0; i < bgImage.rows; ++i) {
	    for(int j = 0; j < bgImage.cols; ++j, ++p) {
		bgImage.at<cv::Vec3b>(i,j) = bgModel[p].values[0];
	    }
	}
		
	if(bgFilename != NULL) {
	    ofstream file;
	    file.open(bgFilename->c_str());
	    file<<(int)frameSize.width<<" ";
	    file<<(int)frameSize.height<<endl;
	    file<<(int)frameType<<endl;
	    int c = 0;
	    for(int i = 0; i<frameSize.height; i++) {
		for(int j = 0; j<frameSize.width; j++, c++) {
		    for(unsigned int e = 0; e < maxBgBins; e++) {
			if(!bgModel[c].isValid[e]) {
			    file<<endl;
			    break;
			}
			file<<(int)bgModel[c].values[e].val[2]<<" ";
			file<<(int)bgModel[c].values[e].val[1]<<" ";
			file<<(int)bgModel[c].values[e].val[0]<<" ";
			if(e == (maxBgBins - 1)) {
			    file<<endl;
			}
		    }
		}
	    }
	    file.close();
	    bgFilename = NULL;
	}//if bgFilename	
				
    }
}

void BackgroundSubtractorIMBS::createIncrementalBg(unsigned int bg_sample_number)
{
    if(!bgSample.data) {
        //cerr << "createBg -- an error occurred: " <<
        //		" unable to retrieve frame no. " << bg_sample_number << endl;

        //TODO vedere gestione errori
        abort();
    }
    Vec3b currentPixel;
    //split bgSample in channels
    cv::split(bgSample, bgSampleBGR);

    //create a statistical model for each pixel (a set of bins of variable size)
    for(unsigned int p = 0; p < numPixels; ++p) {
        //create an initial bin for each pixel from the first sample (bg_sample_number = 0)
        if(bg_sample_number == 0) {

            incrementalBgBins[p].binValues[0][0] = bgSampleBGR[0].data[p];
            incrementalBgBins[p].binValues[0][1] = bgSampleBGR[1].data[p];
            incrementalBgBins[p].binValues[0][2] = bgSampleBGR[2].data[p];

            incrementalBgBins[p].binHeights[0] = 1;
            std::fill(incrementalBgBins[p].binHeights.begin() + 1, incrementalBgBins[p].binHeights.end(), 0);

            //if the sample pixel is from foreground keep track of that situation
            incrementalBgBins[p].isFg[0] = (fgmask.data[p] == FOREGROUND_LABEL) ? true : false;

        }//if(bg_sample_number == 0)
        //if all samples have been processed
        //it is time to compute the fg mask
        else if(bg_sample_number == (incrementalNumSamples - 1)) {
            unsigned int index = 0;
            int max_height = -1;

            for(unsigned int s = 0; s < incrementalNumSamples; ++s){
                if(incrementalBgBins[p].binHeights[s] == 0) {
                    incrementalBgModel[p].isValid[index] = false;
                    break;
                }
                if(index == maxBgBins) {
                    --incrementalCounters[p];
                    break;
                }
                else if(incrementalBgBins[p].binHeights[s] >= minBinHeight) {

                    if(incrementalBgBins[p].binHeights[s] > max_height) {

                        max_height = incrementalBgBins[p].binHeights[s];

                        incrementalBgModel[p].values[index][0] = incrementalBgModel[p].values[0][0];
                        incrementalBgModel[p].values[index][1] = incrementalBgModel[p].values[0][1];
                        incrementalBgModel[p].values[index][2] = incrementalBgModel[p].values[0][2];

                        incrementalBgModel[p].isValid[index] = true;
                        incrementalBgModel[p].isFg[index] = incrementalBgModel[p].isFg[0];
                        incrementalBgModel[p].counter[index] = incrementalBgModel[p].counter[0];

                        incrementalBgModel[p].values[0][0] = incrementalBgBins[p].binValues[s][0];
                        incrementalBgModel[p].values[0][1] = incrementalBgBins[p].binValues[s][1];
                        incrementalBgModel[p].values[0][2] = incrementalBgBins[p].binValues[s][2];
                        
                        incrementalBgModel[p].isValid[0] = true;
                        incrementalBgModel[p].isFg[0] = incrementalBgBins[p].isFg[s];
                        incrementalBgModel[p].counter[0] = incrementalBgBins[p].binHeights[s];
                    }
                    else {

                        incrementalBgModel[p].values[index][0] = incrementalBgBins[p].binValues[s][0];
                        incrementalBgModel[p].values[index][1] = incrementalBgBins[p].binValues[s][1];
                        incrementalBgModel[p].values[index][2] = incrementalBgBins[p].binValues[s][2];

                        incrementalBgModel[p].isValid[index] = true;
                        incrementalBgModel[p].isFg[index] = incrementalBgBins[p].isFg[s];
                        incrementalBgModel[p].counter[index] = incrementalBgBins[p].binHeights[s];
                    }
                    ++incrementalCounters[p];
                    ++index;
                }
            } //for all numSamples
        }//bg_sample_number == (numSamples - 1)
        else { //bg_sample_number > 0 && bg_sample_number != (numSamples - 1)
            currentPixel[0] = bgSampleBGR[0].data[p];
            currentPixel[1] = bgSampleBGR[1].data[p];
            currentPixel[2] = bgSampleBGR[2].data[p];

            int den = 0;
            for(unsigned int s = 0; s < bg_sample_number; ++s) {
                //try to associate the current pixel values to an existing bin
                if( this->abs_(currentPixel[2] - incrementalBgBins[p].binValues[s][2]) <= (int) associationThreshold &&
                         this->abs_(currentPixel[1] - incrementalBgBins[p].binValues[s][1]) <= (int) associationThreshold &&
                         this->abs_(currentPixel[0] - incrementalBgBins[p].binValues[s][0]) <= (int) associationThreshold )
                {
                    den = (incrementalBgBins[p].binHeights[s] + 1);
                    for(int k = 0; k < 3; ++k) {
                        incrementalBgBins[p].binValues[s][k] =
                            (incrementalBgBins[p].binValues[s][k] * incrementalBgBins[p].binHeights[s] + currentPixel[k]) / den;
                    }
                    incrementalBgBins[p].binHeights[s]++; //increment the height of the bin
                    if(fgmask.data[p] == FOREGROUND_LABEL) {
                        incrementalBgBins[p].isFg[s] = true;
                    }
                    break;
                }
                //if the association is not possible, create a new bin
                else if(incrementalBgBins[p].binHeights[s] == 0)	{

                    incrementalBgBins[p].binValues[s] = currentPixel;
                    incrementalBgBins[p].binHeights[s]++;
                    incrementalBgBins[p].isFg[s] = (fgmask.data[p] == FOREGROUND_LABEL) ? true : false;

                    break;
                }
                else continue;
            }//for(unsigned int s = 0; s <= bg_sample_number; ++s)            
        }//else --> //bg_sample_number > 0 && bg_sample_number != (numSamples - 1)
    }//numPixels

    if(bg_sample_number == (incrementalNumSamples - 1)) {
        std::cout << "new bg created" << std::endl;

        //incrementalSamplingPeriod += incrementalSamplingPeriod;

        incrementalNumSamples *= 2;
        if(incrementalNumSamples > numSamples)
        {
            incrementalNumSamples = numSamples;
        }

        counterCopy = incrementalCounters;
        std::fill(incrementalCounters.begin(), incrementalCounters.end(), 0);

        unsigned int p = 0;
        for(int i = 0; i < bgImage.rows; ++i) {
            for(int j = 0; j < bgImage.cols; ++j, ++p) {
                bgImage.at<cv::Vec3b>(i,j) = incrementalBgModel[p].values[0];
            }
        }
    }
}

void BackgroundSubtractorIMBS::getFg() {
	fgmask = Scalar(0);
	cv::split(frame, frameBGR);

	bool isFg = true;
	bool conditionalUpdated = false;
	unsigned int d = 0;
	for(unsigned int p = 0; p < numPixels; ++p) {
		isFg = true;
		conditionalUpdated = false;
		d = 0;
        unsigned int end = counterCopy[p];
        for(unsigned int n = 0; n < end; ++n) {
			if(!bgModel[p].isValid[n]) {
				if(n == 0) {
					isFg = false;
				}
				break;
			}
			else { //the model is valid
                int a = this->abs_(bgModel[p].values[n][0] - frameBGR[0].data[p]);
                int b = this->abs_(bgModel[p].values[n][1] - frameBGR[1].data[p]);
                int c = this->abs_(bgModel[p].values[n][2] - frameBGR[2].data[p]);
                d = this->max_(a, b, c);

                if(d < fgThreshold){
					//check if it is a potential background pixel
					//from stationary object
					if(bgModel[p].isFg[n]) {
						conditionalUpdated = true;
						break;
					}
					else {
						isFg = false;
						persistenceMap[p] = 0;
					}
				}
			}
		}
		if(isFg) {
			if(conditionalUpdated) {
				fgmask.data[p] = PERSISTENCE_LABEL;
				persistenceMap[p] += (timestamp - prev_timestamp);
				if(persistenceMap[p] > persistencePeriod) {
                    for(unsigned int n = 0; n < end; ++n) {
						if(!bgModel[p].isValid[n]) {
							break;
						}
						bgModel[p].isFg[n] = false;
					}
				}
			}
			else {
				fgmask.data[p] = FOREGROUND_LABEL;
				persistenceMap[p] = 0;
			}
		}
    }
}

void BackgroundSubtractorIMBS::getIncrementalFg()
{
    //fgmask = Scalar(0);
    cv::split(frame, frameBGR);

    bool isFg = true;
    bool conditionalUpdated = false;
    unsigned int d = 0;
    for(unsigned int p = 0; p < numPixels; ++p) {
        isFg = true;
        conditionalUpdated = false;
        d = 0;
        unsigned int end = counterCopy[p];
        for(unsigned int n = 0; n < end; ++n) {
            if(!incrementalBgModel[p].isValid[n]) {
                if(n == 0) {
                    isFg = false;
                }
                break;
            }
            else { //the model is valid
                int a = this->abs_(incrementalBgModel[p].values[n][0] - frameBGR[0].data[p]);
                int b = this->abs_(incrementalBgModel[p].values[n][1] - frameBGR[1].data[p]);
                int c = this->abs_(incrementalBgModel[p].values[n][2] - frameBGR[2].data[p]);
                d = this->max_(a, b, c);

                if(d < fgThreshold){
                    //check if it is a potential background pixel
                    //from stationary object
                    if(incrementalBgModel[p].isFg[n]) {
                        conditionalUpdated = true;
                        break;
                    }
                    else {
                        isFg = false;
                    }
                }
            }
        }
        if(isFg) {
            if(conditionalUpdated)
            {
                fgmask.data[p] = PERSISTENCE_LABEL;
            }
            else
            {
                fgmask.data[p] = FOREGROUND_LABEL;
            }
        }
    }
}

void BackgroundSubtractorIMBS::areaThresholding()
{
	double maxArea = 0.6 * numPixels;
	
	std::vector < std::vector<Point> > contours;

    Mat tmpBinaryImage = fgfiltered.clone();
    findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
    Moments moms;
    double area;
    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
    {
        moms = moments(Mat(contours[contourIdx]));
        area = moms.m00;
        if (area < minArea || area >= maxArea)
        {
            drawContours( fgfiltered, contours, contourIdx, Scalar(0), CV_FILLED );
        }
	}	
}

// Create a HSV image from the RGB image using the full 8-bits, since OpenCV only allows Hues up to 180 instead of 255.
// ref: "http://cs.haifa.ac.il/hagit/courses/ist/Lectures/Demos/ColorApplet2/t_convert.html"
// Remember to free the generated HSV image.
void BackgroundSubtractorIMBS::convertImageRGBtoHSV(const Mat& imageRGB)
{
	float fR, fG, fB;
	float fH, fS, fV;

	float fDelta;
	float fMin, fMax;
	int iMax;

	int bB; // Blue component
	int bG; // Green component
	int bR;	// Red component

	int bH;
	int bS;
	int bV;
	
	// Create a blank HSV image
    imageHSV = cv::Mat::zeros(imageRGB.size(), CV_8UC3);
	//if (!imageHSV || imageRGB->depth != 8 || imageRGB->nChannels != 3) {
		//printf("ERROR in convertImageRGBtoHSV()! Bad input image.\n");
		//exit(1);
	//}

	int h = imageRGB.rows;		// Pixel height.
	int w = imageRGB.cols;		// Pixel width.
	//int rowSizeRGB = imageRGB->widthStep;	// Size of row in bytes, including extra padding.
	//char *imRGB = imageRGB->imageData;	// Pointer to the start of the image pixels.
	//int rowSizeHSV = imageHSV->widthStep;	// Size of row in bytes, including extra padding.
	//char *imHSV = imageHSV->imageData;	// Pointer to the start of the image pixels.
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			// Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
			//uchar *pRGB = (uchar*)(imRGB + y*rowSizeRGB + x*3);
			bB = imageRGB.at<Vec3b>(y,x)[0]; //*(uchar*)(pRGB+0);	// Blue component
			bG = imageRGB.at<Vec3b>(y,x)[1]; //*(uchar*)(pRGB+1);	// Green component
			bR = imageRGB.at<Vec3b>(y,x)[2]; //*(uchar*)(pRGB+2);	// Red component

			// Convert from 8-bit integers to floats.
			fR = bR * BYTE_TO_FLOAT;
			fG = bG * BYTE_TO_FLOAT;
			fB = bB * BYTE_TO_FLOAT;

			// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
			
			// Get the min and max, but use integer comparisons for slight speedup.
			if (bB < bG) {
				if (bB < bR) {
					fMin = fB;
					if (bR > bG) {
						iMax = bR;
						fMax = fR;
					}
					else {
						iMax = bG;
						fMax = fG;
					}
				}
				else {
					fMin = fR;
					fMax = fG;
					iMax = bG;
				}
			}
			else {
				if (bG < bR) {
					fMin = fG;
					if (bB > bR) {
						fMax = fB;
						iMax = bB;
					}
					else {
						fMax = fR;
						iMax = bR;
					}
				}
				else {
					fMin = fR;
					fMax = fB;
					iMax = bB;
				}
			}
			fDelta = fMax - fMin;
			fV = fMax;				// Value (Brightness).
			if (iMax != 0) {			// Make sure its not pure black.
				fS = fDelta / fMax;		// Saturation.
				float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta);	// Make the Hues between 0.0 to 1.0 instead of 6.0
				if (iMax == bR) {		// between yellow and magenta.
					fH = (fG - fB) * ANGLE_TO_UNIT;
				}
				else if (iMax == bG) {		// between cyan and yellow.
                    fH = fhV1 + ( fB - fR ) * ANGLE_TO_UNIT;
				}
				else {				// between magenta and cyan.
                    fH = fhV2 + ( fR - fG ) * ANGLE_TO_UNIT;
				}
				// Wrap outlier Hues around the circle.
				if (fH < 0.0f)
					fH += 1.0f;
				if (fH >= 1.0f)
					fH -= 1.0f;
			}
			else {
				// color is pure Black.
				fS = 0;
				fH = 0;	// undefined hue
			}

			// Convert from floats to 8-bit integers.
			bH = (int)(0.5f + fH * 255.0f);
			bS = (int)(0.5f + fS * 255.0f);
			bV = (int)(0.5f + fV * 255.0f);

			// Clip the values to make sure it fits within the 8bits.
			if (bH > 255)
				bH = 255;
			if (bH < 0)
				bH = 0;
			if (bS > 255)
				bS = 255;
			if (bS < 0)
				bS = 0;
			if (bV > 255)
				bV = 255;
			if (bV < 0)
				bV = 0;

			// Set the HSV pixel components.
			imageHSV.at<Vec3b>(y, x)[0] = bH;		// H component
			imageHSV.at<Vec3b>(y, x)[1] = bS;		// S component
			imageHSV.at<Vec3b>(y, x)[2] = bV;		// V component
		}
	}
}

void BackgroundSubtractorIMBS::getBackgroundImage(OutputArray backgroundImage) const
{
    bgImage.copyTo(backgroundImage);        
}

void BackgroundSubtractorIMBS::filterFg() {

    cv::Mat mask = fgmask == 255;
    unsigned int cnt = cv::countNonZero(mask);

    fgfiltered = Scalar(0);
    fgmask.copyTo(fgfiltered, mask);

    if (cnt > numPixels*0.5) {
        incremental_bg = true;
    }

	if(morphologicalFiltering) {
		cv::Mat element3(3,3,CV_8U,cv::Scalar(1));
		cv::morphologyEx(fgfiltered, fgfiltered, cv::MORPH_OPEN, element3);
		cv::morphologyEx(fgfiltered, fgfiltered, cv::MORPH_CLOSE, element3);
	}

    //areaThresholding();

    cv::Mat persistenceMask = fgmask == PERSISTENCE_LABEL;
    cv::Mat shadowMask = fgmask == SHADOW_LABEL;

    fgmask.copyTo(fgfiltered, persistenceMask);
    fgmask.copyTo(fgfiltered, shadowMask);

    fgfiltered.copyTo(fgmask);
}

int BackgroundSubtractorIMBS::abs_(int _value)
{
    int abs_value = _value;
    uint32_t temp = abs_value >> 31;
    abs_value ^= temp;
    abs_value += temp & 1;
    return abs_value;
}

int BackgroundSubtractorIMBS::max_(int _a, int _b, int _c)
{
    int max_value = _a;
    (max_value < _b) && (max_value = _b);
    (max_value < _c) && (max_value = _c);
    return max_value;
}

int BackgroundSubtractorIMBS::min_(int _a, int _b)
{
    return  _b + ((_a - _b) & ((_a - _b) >>
                            (sizeof(int) * CHAR_BIT - 1)));
}

void BackgroundSubtractorIMBS::getBgModel(vector<BgModel> &bgModel_copy)
{
    bgModel_copy = bgModel;
}


bool BackgroundSubtractorIMBS::loadBg(const char* filename) {
	string line;
	ifstream file(filename, ifstream::in);
	int c = 0;
	if(file.is_open()) {
		loadedBg = true;
		
		//initialization step
		cout << endl;
		cout << "LOADED BG" << endl;
		cout << endl;
		
		//get frame size and frame type
		getline(file, line);
		cout << line << endl;
		
		int index = line.find_first_of(" ");
		string widthString = line.substr(0, index);
		int width;
		istringstream ss_w(widthString);
		ss_w >> width;
		line.erase(0, index+1);
		
		string heightString = line;
		int height;
		istringstream ss_h(heightString);
		ss_h >> height;
		
		cout << "width " << width << "   height " << height << endl;
		
		Size frameSize(width, height);
		getline(file, line);
		cout << line << endl;
		int frameType = 0;
		
		cout << "INPUT: WIDTH " << frameSize.width << "  HEIGHT " << frameSize.height <<
			"  FPS " << fps << endl;
		cout << endl;

		this->frameSize = frameSize;
		this->frameType = frameType;
		this->numPixels = frameSize.width*frameSize.height;

        persistenceMap.resize(numPixels, 0);
        counters.resize(numPixels, 0);
        incrementalCounters.resize(numPixels, 0);


        bgBins.resize(numPixels);
        bgModel.resize(numPixels);

        incrementalBgBins.resize(numPixels);
        incrementalBgModel.resize(numPixels);

		maxBgBins = numSamples / minBinHeight;

        timestamp = 0.;//ms
        prev_timestamp = 0.;//ms
        prev_bg_frame_time = 0;
        prev_incremental_bg_frame_time = 0;
        bg_frame_counter = 0;
        incremental_bg_frame_counter = 0;
        prev_area = 0;
        incremental_bg = false;

		SHADOW_LABEL = 80;
		PERSISTENCE_LABEL = 180;
		FOREGROUND_LABEL = 255;

		fgmask.create(frameSize, CV_8UC1);
		fgfiltered.create(frameSize, CV_8UC1);
		persistenceImage = Mat::zeros(frameSize, CV_8UC1);
		bgSample.create(frameSize, CV_8UC3);
		bgImage = Mat::zeros(frameSize, CV_8UC3);

		if(minBinHeight <= 1){
			minBinHeight = 1;
		}

		for(unsigned int p = 0; p < numPixels; ++p)
		{
            bgBins[p].binValues.resize(numSamples);
            bgBins[p].binHeights.resize(numSamples);
            bgBins[p].isFg.resize(numSamples, false);

            bgModel[p].values.resize(maxBgBins);
            bgModel[p].isValid.resize(maxBgBins);
            bgModel[p].isValid[0] = false;
            bgModel[p].isFg.resize(maxBgBins, false);
            bgModel[p].counter.resize(maxBgBins);

            incrementalBgBins[p].binValues.resize(numSamples);
            incrementalBgBins[p].binHeights.resize(numSamples);
            incrementalBgBins[p].isFg.resize(numSamples, false);

            incrementalBgModel[p].values.resize(maxBgBins);
            incrementalBgModel[p].isValid.resize(maxBgBins);
            incrementalBgModel[p].isValid[0] = false;
            incrementalBgModel[p].isFg.resize(maxBgBins, false);
            incrementalBgModel[p].counter.resize(maxBgBins);
		}		
		
		while(!file.eof()) {
			getline(file, line);

			int n = 0;

			while(line.length() > 1){

				int index = line.find_first_of(" ");
				string red = line.substr(0, index);
				int r;
				istringstream ss_r(red);
				ss_r >> r;
				line.erase(0, index+1);

				index = line.find_first_of(" ");
				string green = line.substr(0, index);
				int g;
				istringstream ss_g(green);
				ss_g >> g;
				line.erase(0, index+1);

				index = line.find_first_of(" ");
				string blue = line.substr(0, index);
				int b;
				istringstream ss_b(blue);
				ss_b >> b;
				line.erase(0, index+1);
				
				//cout << "io" << endl;

				bgModel[c].values[n].val[0] = b;
				//cout << "io 1" << endl;
				bgModel[c].values[n].val[1] = g;
				//cout << "io 2" << endl;
				bgModel[c].values[n].val[2] = r;
				//cout << "io 3" << endl;
				bgModel[c].isValid[n] = true;
				//cout << "io 4" << endl;
				bgModel[c].isFg[n] = false;
				//cout << "io 5" << endl;
				bgModel[c].counter[n] = minBinHeight;
				//cout << "io 6" << endl;

				if(n == 0) {
					int i = c/bgImage.cols;
					int j = c - i*bgImage.cols;
					//cout << "i " << i << " j " << j << endl;
					bgImage.at<Vec3b>(i, j) = bgModel[c].values[0];
				}
				n++;
			}
			c++;
	    }
	    file.close();
	    return true;
	}
	else {
		return false;
	}
}


void BackgroundSubtractorIMBS::saveBg(string* filename) {
	bgFilename = filename;
}
