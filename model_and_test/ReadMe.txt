The python files are model(network architecture with training and testing functions) and helper file(early stopping source code) which relocalizes the robot in an indoor environment.
The model.py regresses x, y and the orientation of the robot with respect to the starting pose(reference frame).
The codes written are rapid prototyping codes which work according to the requirements of the project and has the driver function also inside it. The test.py is also embedded inside the files. 
Transfer Learning was used to extract features(ResNet-50) as our dataset did not contain a lot of images for training a network from scratch.
L1 loss is used in the model as the mean absolute deviation loss was more effective than L2 loss for our custom dataset.
pytorchtools.py is a helper file which helps in the implementation of early stopping which was used to overcome overfitting. 

