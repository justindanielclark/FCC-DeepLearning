
#^ Computer Vision

#? Representing Image Data - A 28x28 Image with 3 Color Channels
  #? Shape = [None, 28, 28, 3] - (NHWC) - Number, Height, Width, Color Channels
  #? Shape = [None, 3, 28, 28] - (NCHW) - Number, Color Channels, Height, Width

#^ What is a convolutional neural network

#? Disadvantages of using ANN (Artificial Neural Network) for image classification
  #? Too much computation
  #? Treats loacl pixles the same as pixels far apart
  #? Sensitive to the location of an object in an image
#? Think About how a person recognizes a koala
  #? You First might recognize koala ears / eyes / mouth -> now you are looking at a Koala's head
  #? You might look at its area under its head looking for part of a body. IF it has Koala parts, its a full Koala
#? We are looking for features! For numbers, we are looking for straight lines / loopy patterns
  #? Pattern searches are location invariant!
    #? Now we only care about the relations of these patterns to each other

#^ Benefits of CNN
#? Convolution:
  #? Connections sparsity reduces overfitting
  #? Conv + Pooling gives location invariant feature detection
  #? Parameter sharing
#? ReLU
  #? Introduces nonlinearity
  #? Speeds  up training, faster computes
#? Pooling
  #? Reduces overfitting
  #? Reduces dimensions / computations
  #? Model becomes tolerant to small distortions

#^ CNN Does not take care of rotation or scale
#? You need to have rotated, scaled sampels in training dataset
#? If you do not have such samples, than use data augmentation methods to generate new rotated/scaled samples from existing training samples