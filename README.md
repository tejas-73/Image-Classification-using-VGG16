# Image-Classification-using-VGG16
The goal was to classify images into respective classes. We have considered images of Airplanes, Bikes, Human faces and Cars as our classes 

VGG16 was the feature extractor used with top 3 layers removed.
1)	The dimensions of feature vector were (7, 7, 512) obtained from block_5 pooling layer of VGG16, which was later flattened for comparison purposes.
2)	KMeans clustering was used for image prediction.
3)	The input to the Kmeans algorithm given was such that first few(in our case it was 30) images belong to same class of images(say airplanes), then the other class of images were given sequentially and the mode of the output data from KMeans algorithm was taken for particular interval(in our case, it was 30) and defined to be the class identity for the given class.
4)	The file path was used for identifying the image contents for comparison of the prediction by algorithm.
For instance, images of airplanes were stored in “Images/airplanes_test/img0xx.jpg”
The string “airplanes” present in the file path was used as a justification that the image is of airplane.

Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
Positive Predictions Made by the Algorithm:   80
Negative Predictions Made by the Algorithm:   0
Prediction Accuracy is:   100.0 %
Time Required to Execute programme:   41.272661447525024  Seconds.

Process finished with exit code 0
