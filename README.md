## Face Mask Detection Using MobileNetV2

This a Intelligent Face Mask Detection system to track whether a person is wearing a Face Mask or not in Real Time Environmnent. This model was built using the following packages:
* Tensorflow
* Keras
* Scikit-learn
* OpenCV 

This CNN based model uses the MobileNetV2 Architecture due to it's light weight, fewer parameters and high accuracy. The Real Time environment is built using the OpenCV package. 

## Model Flow
* Input of images is fed into the layer.
*  AveragePooling2D is implemented to reduce the dimensionality.
* The resultant is then flattened into a vector and then sent into a fully connected layer.
* Dropout is implemented to improve the performance. During this few neurons are dropped to prevent the model from overfitting.
* Then an activation function is used and the class label is given as output.

![Web capture_20-10-2022_114145_](https://user-images.githubusercontent.com/86162963/196869857-84198e4e-c4cc-4033-9531-c251e64cfb75.jpeg)

## Results
