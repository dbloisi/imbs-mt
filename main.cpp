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

//C
#include <stdio.h>
//C++
#include <iostream>
//OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "imagemanager.h"

//imbs-mt
#include "imbs.hpp"
#include "imbsmultithread.h"

#define SHADOW_LABEL 80
#define PERSISTENCE_LABEL 180
#define FOREGROUND_LABEL 255

using namespace cv;
using namespace std;

/**
 * Global variables
 */
vector<BackgroundSubtractorIMBS> pIMBS;
Mat frame;     //current frame
Mat fgMask;
double fps;   //frame per second for the input video sequence
int keyboard;  //input from keyboard
int counter = 0;
std::shared_ptr<BackgroundSubtractorIMBSMT> imbsMT;

/**
 * Function Headers
*/
void help();
void processVideo(char* videoFilename, double _fps=-1.);
void processImages(char* firstFrameFilename);

/**
* Structs
*/
struct ThreadData
{
	Mat threadFrame;
	int counter;
};

struct ThreadResult
{
	Mat threadFrame;
	Mat threadBgImage;
	Mat threadFgMask;
};

/**
* @function help
*/
void help()
{
    cout
    << "--------------------------------------------------------------------------" << endl
    << "IMBS Background Subtraction Library "                                       << endl
    << "This file main.cpp contains an example of usage for"                        << endl
    << "IMBS algorithm described in"                                                << endl
    << "D. D. Bloisi and L. Iocchi"                                                 << endl
    << "\"Independent Multimodal Background Subtraction\""                          << endl
    << "In Proc. of the Third Int. Conf. on Computational Modeling of Objects"      << endl
    << "Presented in Images: Fundamentals, Methods and Applications,"               << endl
    << "pp. 39-44, 2012."                                                           << endl
                                                                                    << endl
    << "written by Domenico D. Bloisi"                                              << endl
    << "domenico.bloisi@gmail.com"                                                  << endl
    << "--------------------------------------------------------------------------" << endl
    << "You can process both videos (-vid) and images (-img)."                      << endl
                                                                                    << endl
    << "Usage:"                                                                     << endl
    << "imbs {-vid <video filename>|-img <image filename> [-fps <value>]}"          << endl
    << "for example: imbs -vid video.avi"                                           << endl
    << "or: imbs -img /data/images/1.png"                                           << endl
    << "or: imbs -img /data/images/1.png -fps 7"                                    << endl
    << "--------------------------------------------------------------------------" << endl
                                                                                    << endl;
}

/**
* @function main
*/
int main(int argc, char* argv[])
{

	//print help information
	help();

	//check for the input parameter correctness
	if(argc < 3) {
		cerr <<"Incorrect input list" << endl;
		cerr <<"exiting..." << endl;
		return EXIT_FAILURE;
	}
		
	if(strcmp(argv[1], "-vid") == 0) {
		//input data coming from a video
		if (argc > 4) {
			if (strcmp(argv[3], "-fps") == 0) {
				fps = atof(argv[4]);
			}
			else {
				fps = 25.;
			}
			processVideo(argv[2], fps);
		}
		else {
			processVideo(argv[2]);
		}
		
	}
    else if(strcmp(argv[1], "-img") == 0) {
		//input data coming from a sequence of images
		if (argc > 4) {
			if (strcmp(argv[3], "-fps") == 0) {
				fps = atof(argv[4]);
			}
			else {
				fps = 25.;
			}
		}
		else {
			fps = 25.;
		}
		processImages(argv[2]);
    }
	else {
		//error in reading input parameters
		cerr <<"Please, check the input parameters." << endl;
		cerr <<"Exiting..." << endl;
		return EXIT_FAILURE;
	}
	//destroy GUI windows
	destroyAllWindows();
	return EXIT_SUCCESS;
}

/**
* @function processVideo
*/
void processVideo(char* videoFilename, double _fps) {
    //create the capture object
    VideoCapture capture(videoFilename);
    if(!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }	
	if(_fps < 0.)    
		fps = capture.get(5); //CV_CAP_PROP_FPS
	else
		fps = _fps;
	
    if(fps != fps) { //check for nan value
		fps = 25.;
	}	
	//std::cout << "Number of cores:" << std::thread::hardware_concurrency() << std::endl;
	
    imbsMT = std::shared_ptr<BackgroundSubtractorIMBSMT>(
		new BackgroundSubtractorIMBSMT(
			std::thread::hardware_concurrency(), //numberOfCores
			fps, //frames per second [default 25]
			20, //fgThreshold [default 20]
			5,  //associationThreshold [default 5]
			500., //samplingPeriod [default 500.]
			2,   //minBinHeight [default 2]
			30, //numSamples [default 20]
			0.65, //alpha = default [0.65]
			1.15, //beta [default 1.15]
			60., //tau_s = [default 60.]
			40., //tau_h [default 40.]
			50., //minArea = [default 50.]
			10000., //persistencePeriod [default 10000.]
			false //morphologicalFiltering = [default false]
		)
	);	

	//read input data. ESC or 'q' for quitting
	int sleepTime = 30;
    while( (char)keyboard != 'q' && (char)keyboard != 27 ){
		
		//read the current frame
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;		
            exit(EXIT_FAILURE);
        }

        imbsMT->apply(frame, fgMask);       

		stringstream stream;
		rectangle(frame, cv::Point(10, 2), cv::Point(100,20),cv::Scalar(255,255,255), -1);
        stream << capture.get(1);
		string frameNumberString = stream.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
		
		//show the current frame and the fg masks
        imshow("Input Frame", frame);
        imshow("FgMask", fgMask);
        imshow("BgModel", imbsMT->getBgModel());
        		
		keyboard = waitKey(sleepTime);
    }
    //delete capture object
    capture.release();
}

/**
* @function processImages
*/
void processImages(char* firstFrameFilename) {	

	string foldername(firstFrameFilename);
	size_t folder_index = foldername.find_last_of("/");
    if(folder_index == string::npos) {
    	folder_index = foldername.find_last_of("\\");
    }
    foldername = foldername.substr(0,folder_index+1);
    
	ImageManager *im = new ImageManager(foldername);

    //read the first file of the sequence
	string s = im->next(1);
	size_t index = s.find_last_of("/"); 
	if (index != string::npos) {
		s.erase(s.begin() + index);
	}

    //frame = imread(foldername+s);
	frame = imread(s);

    if(!frame.data){
        //error in opening the first image
        cerr << "Unable to open first image frame: " << s << endl;
        exit(EXIT_FAILURE);
    }
	
    imbsMT = std::shared_ptr<BackgroundSubtractorIMBSMT>(		
		new BackgroundSubtractorIMBSMT(
			std::thread::hardware_concurrency(), //numberOfCores
			fps, //frames per second [default 25]
			20, //fgThreshold [default 20]
			5,  //associationThreshold [default 5]
			500., //samplingPeriod [default 500.]
			2,   //minBinHeight [default 2]
			30, //numSamples [default 20]
			0.65, //alpha = default [0.65]
			1.15, //beta [default 1.15]
			60., //tau_s = [default 60.]
			40., //tau_h [default 40.]
			50., //minArea = [default 50.]
			10000., //persistencePeriod [default 10000.]
			false //morphologicalFiltering = [default false]
		)		
	);

    int frameRateCounter = 0;

    //read input data. ESC or 'q' for quitting
	int sleepTime = 30;
    while( (char)keyboard != 'q' && (char)keyboard != 27 ){
    	//update the background model
        imbsMT->apply(frame, fgMask);        

        ++frameRateCounter;

        stringstream stream;
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),cv::Scalar(255,255,255), -1);
        stream << frameRateCounter;
        string frameNumberString = stream.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));


        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        //show the current frame and the fg masks
        imshow("Input Frame", frame);
        imshow("FG Mask", fgMask);
        imshow("BgModel", imbsMT->getBgModel());

		keyboard = waitKey(sleepTime);
        
        //read next frame
        s = im->next(1);
		index = s.find_last_of("/");
		if (index != string::npos) {
			s.erase(s.begin() + index);
		}

        frame = imread(s);
        if(!frame.data){
            //error in opening the next image in the sequence
            cerr << "Unable to open image frame: " << s << endl;
            exit(EXIT_FAILURE);
        }        
    }	
}
