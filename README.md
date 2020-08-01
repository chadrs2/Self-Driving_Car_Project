# Self-Driving Car Project with a Raspberry Pi using the Raspian OS
![Front of Car](/readme_assets/IMG_6002.jpg) 
![Car Body](/readme_assets/IMG_4112.JPG)
This project contains code that can be implemented on a Raspberry Pi using the Python 3 coding language. This code is designed to be implemented on an already built car with 1 Raspberry Pi, running the newest Raspian OS, as the single-board computer controlling the car. See the **Objectives** point below to see what the code can make the car do.

Videos On Code and Results Found - https://www.youtube.com/channel/UCKXwdkUKI9yCJQC7W0Y0AXw

## Supplies (the ones I used in this project)
* Raspberry Pi 4 Model B 2019 Quad Core 64 Bit WiFi Bluetooth (4GB)
* Raspberry Pi 4 Case w/cooling Fan and Heatsinks
* Pi Camera Wide Angled Fisheye Lens 5MP 1080P
* Micro Servo SG90 9g
* 3V DC Motors (2x)
* L298N Motor Driver
* 64 GB Micro SD Card
* Portable 3A High-Speed Power Bank
* Rechargeable 9V Batteries
* Mini Breadboard
* 470 Ohm Resistors (1x) (more to come once additional future objectives are achieved)
* _Personally Designed and Printed 3D Car Body Parts (can't purchase)_

## Objectives
### Achieved Objectives
* Detect lane lines
  * Code: **DetectLaneLines.py**
* Detect stop sign and stop car 
  * Code: **DetectStopSign_StopCar.py**
  * Data: **stopsign_good.xml**
* Train car on built track using 1 Raspberry Pi camera as the only sensor
  * Code: 
    * To drive car from computer and collect data: **DriveCar_RecordData_Threading.py**
    * To pull data from github and build the CNN model: **BuildCNNModel.py**
    * _Note: **model.h5** and **model.tflite** are my models built from my data_
* Drive car from the CNN (convolutional neural network) model built based on training data in previous step
  * Code: **DriveAutonomously.py**

### Future Objectives
_Note: As these objectives are accomplished, the related code will be added to this repository and these points will be moved to the **Achieved Objectives** section_

* Detect a traffic light at an intersection and respond accordingly
* Stop and avoid humans and other cars on crosswalks or on road
* Implement LiDAR sensor to improve human and other car interaction/path planning
* Add lights and blinkers to drive car in dark areas
