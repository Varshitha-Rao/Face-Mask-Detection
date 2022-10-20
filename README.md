## Face Mask Detection Using MobileNetV2

This a Intelligent Face Mask Detection system to track whether a person is wearing a Face Mask or not in Real Time Environmnent. This model was built using the following packages:
* Tensorflow
* Keras
* Scikit-learn
* OpenCV 

## Dataset
The dataset used is from Kaggle. The Dataset consists of Images:
* With Mask
* WIthout Mask

Then the Datset is split into 
* Training Data
* Validation Data

## Implementation
We have used the Convolutional Neural Networks to train the model. This CNN based model uses the MobileNetV2 Architecture due to it's light weight, fewer parameters and high accuracy. And then the Real Time environment is built using the OpenCV package. 

## Model Flow
We have implemented the MobileNetV2 architecture to ensure accurate face-mask detection. It is an inverted residual system created specifically for mobile devices. A convolutional layer and 19 residual bottleneck layers make up its architecture.

* Input of images is fed into the layer.
*  AveragePooling2D is implemented to reduce the dimensionality.
* The resultant is then flattened into a vector and then sent into a fully connected layer.
* Dropout is implemented to improve the performance. During this few neurons are dropped to prevent the model from overfitting.
* Then an activation function is used and the class label is given as output.

![Web capture_20-10-2022_114145_](https://user-images.githubusercontent.com/86162963/196869857-84198e4e-c4cc-4033-9531-c251e64cfb75.jpeg)

## Results
Our model has a recall of 0.99, it correctly identifies 99 percent of all face mask images and 99 percent of all the without mask images.

![Web capture_20-10-2022_11453_](https://user-images.githubusercontent.com/86162963/196870379-36078760-09fc-4278-b53f-ce0340d79d2a.jpeg)

## Real Time Detection
After detecting the locations of the face and predicting the mask, we use the inbuilt putText method from openCV to display MASK or NO MASK. Similarly, the name of the person if detected will be displayed.

![download](https://user-images.githubusercontent.com/86162963/196871502-460df1ae-4ae9-4e22-b2b1-cfe64ef33298.jpg)
