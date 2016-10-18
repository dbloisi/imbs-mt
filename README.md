# Independent Multimodal Background Subtraction multi-thread
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

IMBS-MT works under Linux, Mac Os and Windows environments. Please use the following command sequence to build 
the library:

### Linux

* mkdir build
* cd build
* cmake ../
* make -j\<number-of-cores+1\>

### Windows
* Use the CMake graphical user interface to create the desired makefile

## How to use

IMBS-MT is provided with an usage example (main.cpp)

### Linux

For video files:

_$./imbs -vid data/video.avi_

For an image sequence (fps = 25 default value)

_$./imbs -img data/0.jpg_

or you can specify the fps value

_$./imbs -img data/0.jpg -fps 7_


### Windows

For video files

_>imbs -vid data\video.avi_

For an image sequence (fps = 25 default value)

_>imbs -img data\0.jpg_

or you can specify the fps value

_>imbs -img data\0.jpg -fps 7_

For more information on IMBS-MT, you can visit the following webpage: [here](http://www.dis.uniroma1.it/~bloisi/sw/imbs-mt.html).