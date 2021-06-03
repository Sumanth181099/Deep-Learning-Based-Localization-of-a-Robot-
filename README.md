# Deep-Learning-Based-Localization-of-a-Robot-
Autonomous Systems involves three main aspects: Perception, Action/Navigation and control of the system under inspection. 
In this work, we have tackled the perception and control of an autonomous robot working in an indoor environment.
This repo consists of the deep learning based robot localization implementation done using the pytorch deep learning framework. 
The data used here is a custom dataset that was generated using the ROS framework and the Gazebo simulator.
![image](https://user-images.githubusercontent.com/65185434/120710044-4df95580-c4db-11eb-9dcf-973a2b4562c4.png)
The final fully connected layer of the Resnet-50 is deleted and a new fully connected layer is fit with input features being = 2048 and the output being = 1024.
More info are provided inside the ReadMe files inside the specific folders.
