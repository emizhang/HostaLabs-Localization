# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn

# Include any dependencies generated for this target.
include dnn/CMakeFiles/example_dnn_torch_enet.dir/depend.make

# Include the progress variables for this target.
include dnn/CMakeFiles/example_dnn_torch_enet.dir/progress.make

# Include the compile flags for this target's objects.
include dnn/CMakeFiles/example_dnn_torch_enet.dir/flags.make

dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o: dnn/CMakeFiles/example_dnn_torch_enet.dir/flags.make
dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o: torch_enet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o"
	cd /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/dnn && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o -c /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/torch_enet.cpp

dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.i"
	cd /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/dnn && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/torch_enet.cpp > CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.i

dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.s"
	cd /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/dnn && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/torch_enet.cpp -o CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.s

dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.requires:

.PHONY : dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.requires

dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.provides: dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.requires
	$(MAKE) -f dnn/CMakeFiles/example_dnn_torch_enet.dir/build.make dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.provides.build
.PHONY : dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.provides

dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.provides.build: dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o


# Object files for target example_dnn_torch_enet
example_dnn_torch_enet_OBJECTS = \
"CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o"

# External object files for target example_dnn_torch_enet
example_dnn_torch_enet_EXTERNAL_OBJECTS =

dnn/example_dnn-torch_enet: dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o
dnn/example_dnn-torch_enet: dnn/CMakeFiles/example_dnn_torch_enet.dir/build.make
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_stitching.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_superres.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_videostab.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_aruco.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_bgsegm.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_bioinspired.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_ccalib.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_dpm.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_face.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_freetype.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_fuzzy.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_hdf.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_img_hash.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_line_descriptor.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_optflow.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_reg.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_rgbd.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_saliency.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_stereo.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_structured_light.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_surface_matching.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_tracking.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_xfeatures2d.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_ximgproc.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_xobjdetect.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_xphoto.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_shape.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_photo.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_calib3d.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_phase_unwrapping.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_video.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_datasets.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_plot.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_text.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_dnn.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_features2d.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_highgui.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_videoio.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_flann.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_ml.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_imgcodecs.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_objdetect.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_imgproc.so.3.3.1
dnn/example_dnn-torch_enet: /home/bill/anaconda3/envs/my_env/lib/libopencv_core.so.3.3.1
dnn/example_dnn-torch_enet: dnn/CMakeFiles/example_dnn_torch_enet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_dnn-torch_enet"
	cd /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/dnn && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_dnn_torch_enet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dnn/CMakeFiles/example_dnn_torch_enet.dir/build: dnn/example_dnn-torch_enet

.PHONY : dnn/CMakeFiles/example_dnn_torch_enet.dir/build

dnn/CMakeFiles/example_dnn_torch_enet.dir/requires: dnn/CMakeFiles/example_dnn_torch_enet.dir/torch_enet.cpp.o.requires

.PHONY : dnn/CMakeFiles/example_dnn_torch_enet.dir/requires

dnn/CMakeFiles/example_dnn_torch_enet.dir/clean:
	cd /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/dnn && $(CMAKE_COMMAND) -P CMakeFiles/example_dnn_torch_enet.dir/cmake_clean.cmake
.PHONY : dnn/CMakeFiles/example_dnn_torch_enet.dir/clean

dnn/CMakeFiles/example_dnn_torch_enet.dir/depend:
	cd /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/dnn /home/bill/HostaLabs-MachineLearning/CVImageRecognition/KerasSegmentation/opencv/samples/dnn/dnn/CMakeFiles/example_dnn_torch_enet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dnn/CMakeFiles/example_dnn_torch_enet.dir/depend

