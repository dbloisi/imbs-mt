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

#include "imagemanager.h"
#include "natural_less.h"
#include <assert.h>
#if defined(_MSC_VER)
#include <windows.h>
#endif

ImageManager::ImageManager(const std::string &d)
{
    dir_name = d;

#if defined(_MSC_VER)

	WIN32_FIND_DATA ffd;
	LARGE_INTEGER filesize;
	TCHAR szDir[MAX_PATH];
	size_t length_of_arg;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError = 0;
	
	// Check that the input path plus 3 is not longer than MAX_PATH.
	// Three characters are for the "\*" plus NULL appended below.

	StringCchLength(dir_name.c_str(), MAX_PATH, &length_of_arg);

	if (length_of_arg > (MAX_PATH - 3))
	{
		_tprintf(TEXT("\nDirectory path is too long.\n"));
		exit(EXIT_FAILURE);
	}

	_tprintf(TEXT("\nTarget directory is %s\n\n"), dir_name.c_str());

	// Prepare string for use with FindFile functions.  First, copy the
	// string to a buffer, then append '\*' to the directory name.

	StringCchCopy(szDir, MAX_PATH, dir_name.c_str());
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	// Find the first file in the directory.

	hFind = FindFirstFile(szDir, &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		exit(EXIT_FAILURE);
	}

	// List all the files in the directory with some info about them.

	do
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			continue;
        }
		else
		{
			PTSTR pszFileName = ffd.cFileName;
			std::string n(pszFileName);
			filename.push_back(dir_name + "\\" + n);
		}
   }
	while (FindNextFile(hFind, &ffd) != 0);

   dwError = GetLastError();
   if (dwError != ERROR_NO_MORE_FILES)
   {
	   exit(EXIT_FAILURE);
   }

   FindClose(hFind);
   	
#elif defined(__GNUC__) || defined(__GNUG__)
    DIR *dir;
    try
    {
        dir = opendir(d.c_str());
    } catch (std::exception& e)
    {
        std::cout << "Directory does not exist " << e.what() << std::endl;
		exit(EXIT_FAILURE);
    }

    struct dirent *dp;
    while ((dp=readdir(dir)) != NULL) {
        if(strcmp(dp->d_name, "..") != 0  &&  strcmp(dp->d_name, ".") != 0 && dp->d_name[0] != '.'
                && dp->d_name[0] != '~') {
            filename.push_back(dir_name + "/" + std::string(dp->d_name));
        }
    }
#endif

    assert(filename.size() != 0);

    sorting(filename);

    count = -1;
    end = filename.size();
}

ImageManager::~ImageManager() {
    filename.clear();
}

void ImageManager::sorting(std::vector<std::string>& data)
{
    std::sort(data.begin(), data.end(), natural_sort);
}

std::string ImageManager::next(const int &speed) {
    if(count + speed >= end - 1)
    {
        count = end - 1;
    }
    else
    {
        count += speed;
    }
    return filename[count];
}

std::string ImageManager::prev(const int &speed) {
    if(count - speed >= 0)
    {
       count -= speed;
    }
    else
    {
       count = 0;
    }
    return filename[count];
}
