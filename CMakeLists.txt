cmake_minimum_required(VERSION 3.10.0)
project(aitracker VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc video videoio tracking dnn)

# Source files
set(SOURCES
    src/main.cpp
    src/application.cpp
    src/camera_motion.cpp
    src/object_motion.cpp
    src/kalman_predictor.cpp
    src/visualization.cpp
    src/nanotracker.cpp
    src/detector.cpp
    src/tracker_manager.cpp
)

add_executable(aitracker ${SOURCES})
target_include_directories(aitracker PRIVATE ${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aitracker PRIVATE ${OpenCV_LIBS})