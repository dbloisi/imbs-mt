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

#include "imbsmultithread.h"

BackgroundSubtractorIMBSMT::BackgroundSubtractorIMBSMT()
{
    m_bgSubtractors.resize(m_num_cores);
    for(int i = 0; i < m_num_cores; ++i)
    {
        m_bgSubtractors[i] = BackgroundSubtractorIMBS();
    }

    m_horizontalSplits = ceil(m_num_cores * 0.5);

    m_verticalSplits = m_num_cores / m_horizontalSplits;
	
    m_minArea = 50.;
}

BackgroundSubtractorIMBSMT::BackgroundSubtractorIMBSMT(const int &num_cores, const double &fps, const unsigned int &fgThreshold, const unsigned int &associationThreshold, const double &samplingPeriod, const unsigned int &minBinHeight, const unsigned int &numSamples, const double &alpha, const double &beta, const double &tau_s, const double &tau_h, const double &minArea, const double &persistencePeriod, const bool &morphologicalFiltering)
    : m_num_cores(num_cores)
{
    m_bgSubtractors.resize(m_num_cores);
    for(int i = 0; i < m_num_cores; ++i)
    {
        m_bgSubtractors[i] = BackgroundSubtractorIMBS(fps, fgThreshold, associationThreshold, samplingPeriod, minBinHeight, numSamples, alpha, beta, tau_s, tau_h, minArea, persistencePeriod, morphologicalFiltering);
    }

    m_horizontalSplits = ceil(m_num_cores * 0.5);

    m_verticalSplits = m_num_cores / m_horizontalSplits;

    m_minArea = minArea;
}

void BackgroundSubtractorIMBSMT::apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate)
{
    m_frame = image.getMat();
    m_fgMask = cv::Mat(m_frame.size(), CV_8UC1);
    m_bgModel = cv::Mat(m_frame.size(), m_frame.type());

    m_imbsThreads.clear();
    m_imbsThreads.resize(m_num_cores);
    int verticalOffset = m_frame.rows / m_verticalSplits;
    int horizontalOffset = m_frame.cols / m_horizontalSplits;

    m_numPixels = verticalOffset*horizontalOffset;

    std::vector<cv::Mat> splittedFrame = splitFrame(m_frame, horizontalOffset, verticalOffset);
    std::vector<cv::Mat> splittedFgMask = splitFrame(m_fgMask, horizontalOffset, verticalOffset);
    std::vector<cv::Mat> splittedBgModel = splitFrame(m_bgModel, horizontalOffset, verticalOffset);

    int i = 0;
    for(auto &it : m_imbsThreads)
    {
		//cout << "thread " << i << endl;
        it = std::thread(std::bind(imbsThread, std::ref(m_bgSubtractors[i]), learningRate, splittedFrame[i],
                                    std::ref(splittedFgMask[i]), std::ref(splittedBgModel[i])));
        ++i;
    }


    for(auto &it : m_imbsThreads)
    {
        it.join();
    }

    m_fgMask.copyTo(fgmask);

}

std::vector<cv::Mat> BackgroundSubtractorIMBSMT::splitFrame(cv::Mat &frame, const int &hOffset, const int &vOffset)
{
    std::vector<cv::Mat> splittedFrames;
    int beginVertical = 0, endVertical = 0;
    cv::Rect splitWindow;

    for (int i = 0; i < m_verticalSplits; ++i)
    {
        int beginHorizontal = 0, endHorizontal = 0;

        endVertical += vOffset;

        for (int j = 0; j < m_horizontalSplits; ++j)
        {
            endHorizontal += hOffset;

            splitWindow = cv::Rect(beginHorizontal, beginVertical, endHorizontal - beginHorizontal, endVertical - beginVertical);

            splittedFrames.push_back(frame(splitWindow));

            beginHorizontal = endHorizontal;
        }

        beginVertical = endVertical;
    }

    return splittedFrames;
}

const void BackgroundSubtractorIMBSMT::areaThresholding()
{
    double maxArea = 0.6 * m_numPixels;

    std::vector < std::vector<Point> > contours;

    Mat tmpBinaryImage = m_fgMask.clone();
    findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
    Moments moms;
    double area;
    for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
    {
        moms = moments(Mat(contours[contourIdx]));
        area = moms.m00;
        if (area < m_minArea || area >= maxArea)
        {
            drawContours( m_fgMask, contours, contourIdx, Scalar(0), CV_FILLED );
        }
    }
}

void BackgroundSubtractorIMBSMT::imbsThread(BackgroundSubtractorIMBS &imbs, const int &learningRate, cv::Mat &frame, cv::Mat &fg, cv::Mat &bg)
{
	cv::Mat fgMask;
    //get the fgmask and update the background model
    
	imbs.apply(frame, fgMask, learningRate);
    imbs.getBackgroundImage(bg);


    fgMask.copyTo(fg);
}


