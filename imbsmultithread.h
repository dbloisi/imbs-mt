#ifndef IMBSMULTITHREAD_H
#define IMBSMULTITHREAD_H

#include <iostream>
#include <thread>
#include <functional>
#include <opencv2/opencv.hpp>

//imbs
#include "imbs.hpp"


class BackgroundSubtractorIMBSMT
{
    public:
        BackgroundSubtractorIMBSMT();

        BackgroundSubtractorIMBSMT(const int &num_cores,
                                   const double &fps,
                                   const unsigned int &fgThreshold=20,
                                   const unsigned int &associationThreshold=5,
                                   const double &samplingPeriod=500.,
                                   const unsigned int &minBinHeight=2,
                                   const unsigned int &numSamples=20,
                                   const double &alpha=0.65,
                                   const double &beta=1.15,
                                   const double &tau_s=60.,
                                   const double &tau_h=40.,
                                   const double &minArea=50.,
                                   const double &persistencePeriod=10000.,
                                   const bool &morphologicalFiltering=false);

        //! the update operator
        void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate=-1.);
        const inline cv::Mat getBgModel() const&
        {
            return m_bgModel;
        }

    private:
        std::vector<cv::Mat> splitFrame(cv::Mat& frames, const int &hOffset, const int &vOffset);
        const void areaThresholding();
        static void imbsThread(BackgroundSubtractorIMBS &imbs, const int &learningRate, cv::Mat &frame, cv::Mat &fg, cv::Mat &bg);
    private:
        std::vector<BackgroundSubtractorIMBS> m_bgSubtractors;
        std::vector<std::thread> m_imbsThreads;
        int m_num_cores;
        cv::Mat m_frame;
        cv::Mat m_fgMask;
        cv::Mat m_bgModel;
        int m_horizontalSplits;
        int m_verticalSplits;
        int m_numPixels;
        int m_minArea;

};


#endif
