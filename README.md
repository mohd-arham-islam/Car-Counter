# Highway Car Counter using YOLO Algorithm

This repository contains code and instructions for implementing a car counting system using the YOLO (You Only Look Once) algorithm. The project is designed to count the number of cars passing through a specific point on a highway in a video stream. The main steps of the project include preprocessing, object detection, and counting.

![image](https://github.com/mohd-arham-islam/Car-Counter/assets/111959286/35c13fe4-1ea5-4b08-953f-2d42ef8e9675)

## Table of Contents
* **Introduction**
* **Project Overview**
* **Installation**
* **Usage**
* **Results**
* **Contributing**

## Introduction
The goal of this project is to accurately count the number of cars passing through a predetermined point on a highway using the YOLO algorithm. This README provides a comprehensive guide to setting up and running the car counting system on your own.

## Project Overview
The project is divided into three main steps:

**1. Preprocessing**
In this step, a region of interest (ROI) is defined by manually selecting the portion of the highway lane that needs to be analyzed. A binary mask is then applied to the selected ROI, creating a masked image. This masked image is overlaid on each frame of the video using OpenCV. This ensures that only the relevant portion of the video is analyzed by the YOLO model.

**2. Object Detection**
YOLO is employed for object detection. The algorithm detects and localizes cars within each frame of the video. YOLO outputs bounding boxes and confidence scores for detected objects. Non-car objects and objects with low confidence scores are filtered out.

**3. Counting**
The 'sort' library is utilized to assist with car counting. Each detected car is assigned a unique ID, ensuring accurate counting even if the model momentarily fails to identify a car in a few frames. To count the passing cars, a horizontal line is defined across the lane, and each bounding box crossing this line from bottom to top is counted.

## Installation
To set up the car counting system, follow these steps:

**Clone this repository to your local machine using the following command:**

```bash
https://github.com/mohd-arham-islam/Car-Counter.git
```
**Install the required dependencies using the following command:**

```
pip install -r requirements.txt
```

**Run the car counting system using the following command:**

```
python CarCount.py
```

## Results
Upon running the car counting system, the program will analyze the video footage, detect cars, and count their passage through the specified point on the highway. The results will be displayed on the terminal, and a visualized output video will be saved in the output directory.

## Contributing
Contributions to this project are welcome. Feel free to open issues, submit pull requests, or suggest improvements.





