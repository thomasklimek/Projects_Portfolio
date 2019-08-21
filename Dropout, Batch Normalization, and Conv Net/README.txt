Assignment 2: Convolutional Neural Network, Dropout, and Batch Normalization


In this assignment, you will get a deeper understanding of CNN, dropout, and batch normalization. 

There are three main tasks in the three notebook files. In the first task, you need to implement the dropout operation and use it in a feed-forward neural network. In the first step, you need to implement the dropout operation with numpy and make sure your implementation matches the corresponding tensorflow operation, tf.nn.dropout. Then you need to apply the technique to a feedforward neural network by adding dropout layers to the network. You should use tf.nn.dropout in this step -- if you don't want to calculate gradient for your own numpy implementation. An implementation of the feedforward neural network is provided in the handout. You can choose to use this one or use your own implementations. You can modify to the interface (argument lists to the class constructor, train function, and predict function) of the network in the notebook if necessary. 

In the second task, you need to implement the batch normalization operation with numpy. In the first step, you need to implement the batch normalization operation and match the tensorflow layer tf.layers.batch_normalization. Please read the documentation of the tf function carefully so you know what the function actually does, what arguments it needs, and how to use it. In the second step, you need to put tf.batch_normalization to the feedforward neural network and see how it changes the training behavior of the network.

In the third task, you need to implement the convolution operation and the pooling operation and then a convolutional neural network. In the first step, you need to implement the two operations so your implementation matches the calculation of tensorflow operations. In the second step, you need to implement a convolutional neural network and use it for CIFAR10 classification. You may want to modify the feed-forward neural network to make a convolutional one. 

BREAKDOWN of points: 

Implementat dropout (10 points)
Use dropout the neural network  (10 points)

Implementation batch normalization (15 points)
Use batch normalization in the neural network  (15 points)

Implement the convolution operation (20 points)
Implement the convolution neural network and classification (30 points)


DISCLAIMER: the provided implementation of feed-forward neural network may contain errors. If so, you are responsible to correct these errors.    




