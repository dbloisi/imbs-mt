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

/**
 * \file imagemanager.h
 *
 * \class ImageManager
 *
 * \brief Class for managing a set of images
 *
 **/

#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H

#include <iostream>
#include <vector>
#include <string.h>
#include <algorithm>
#include <stdio.h>

#if defined(_MSC_VER)
#include <tchar.h>
#include <strsafe.h>
#pragma comment(lib, "User32.lib")
#elif defined(__GNUC__) || defined(__GNUG__)
#include <dirent.h>
#endif

class ImageManager
{
    public:
        /**
        * \brief Create a new object ImageManager
        *
        * \param d: directory path
        *
        */
        ImageManager(const std::string &d);
        /**
         * \brief ImageManager Destructor
         *
         */
        ~ImageManager();
        /**
         * \brief Return the number of analysed images
         *
         */
        inline int getCount() const
        {
            return count;
        }
        /**
         * \brief Return the end of the image set
         *
         */
        inline int getEnd() const
        {
            return end;
        }
        /**
         * \brief Return the previous image
         *
         * \return return the previous image
         *
         */
        std::string next(const int &speed);
        /**
         * \brief Return current frame number
         *
         * \return return current frame number
         *
         */
        std::string prev(const int &speed);
    private:
        /**
         * \brief Sort the filenames according to the natural sort algorithm
         *
         * \param data: the vector containing the names of the files
         *
         */
        void sorting(std::vector<std::string>& data);
        int count, end;
        std::vector<std::string> filename;
        std::string dir_name;
};

#endif // IMAGEMANAGER_H
