# Independent Multimodal Background Subtraction
Independent Multimodal Background Subtraction multi-thread (IMBS-MT) Library is a C++ library designed for
 performing an accurate foreground extraction in real-time. IMBS creates a multimodal model
of the background in order to deal with illumination changes, camera jitter, movements of 
small background elements, and changes in the background geometry. A statistical analysis 
of the frames in input is performed to obtain the background model. Bootstrap is required 
in order to build the initial background model. IMBS exploits OpenCV functions.

![Example 1](images/bgs-example-1.jpg)
![Example 2](images/bgs-example-2.jpg)

## Requirements

IMBS requires the following packages to build:

* OpenCV (< 3.0)
* C++11

## How to build

IMBS works under Linux, Mac Os and Windows environments. We recommend a so-called out of source build 
which can be achieved by the following command sequence:

### Linux

* mkdir build
* cd build
* cmake ../
* make -j\<number-of-cores+1\>

### Windows
* Use the CMake graphical user interface to create the desired makefile

## How to use

IMBS is provided with an usage example (main.cpp)

### Linux

For video files:

_$./imbs -vid video1.avi_

For an image sequence (fps = 25 default value)

_$./imbs -img /path/to/image/folder_

or you can specify the fps value

_$./imbs -img /path/to/image/folder -fps 7_


### Windows

For video files

_>imbs -vid video1.avi_

For an image sequence (fps = 25 default value)

_>imbs -img \path\to\image\folder_

or you can specify the fps value

_>imbs -img i\path\to\image\folder -fps 7_

For more information, you can visit the following link: [here](http://www.dis.uniroma1.it/~bloisi/software/imbs.html).