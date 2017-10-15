# CCNLSTM
CNN with LSTM to predict next frame

# Starting 
We will utilise the basic Keras Conv_LSTM example and build it. This example creates 1200 15 frame animations of squares moving in a 40x40 window. The basic model is then trained on these examples. Once the training is complete we feed the network the first 7 frames from one of the animations and have it predict the next 8 frames. A comparison is done between the true animation and the one created by the model for those 8 frames.

# Heading to
We will branch off and create our own inputs to the model, they will at first be images resized to 40x40 and be 15frames long for each animation. This should make running them through the model relatively easy. Over time we will resize them to be larger as the assumption would be the cnn can get better information from larger images and hopefully make better guesses for the next frame. 

My plan is to have fun with this and perhaps use short 15frame anime clips. Perhaps during powerup or fighting scenes. We will see what the model predicts will happen next:)

