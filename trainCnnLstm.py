from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import pylab as plt

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3),
                   input_shape=(None, 40, 40, 3), #Will need to change channels to 3 for real images
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(3,3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(Conv3D(filters=3, kernel_size=(3,3,3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')


# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.
### We do not need this function when we create our own animations from images!!

def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 8)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

###OK so lets now create our own set of animations to train on,
### They need to be in format (num_samples, num_frames, rows, cols, channels)

#-----------------WORKING CODE FOR IMPORTING ONE PICTURE AT A TIME, WILL CHANGE TO DIRECTORY ITERATION BELOW-----------#
#movies_input = []
#movies_input_shifted = []
#for i in range(1):
#    movies_input_delayed = []
#    movies_input_shifted_delayed = []
#    for i in range(3):
#        img_path = 'C:\\Users\\DanJas\\Desktop\\CNNLSTM\\Elephant\\elephant' + str(i) + '.jpg'
#        img = load_img(img_path, target_size=(40, 40))
#        x = img_to_array(img)
#        movies_input_delayed.append(x)
#    movies_input.append(movies_input_delayed)
#    for i in range(3):
#        img_path = 'C:\\Users\\DanJas\\Desktop\\CNNLSTM\\Elephant\\elephant' + str(i+1) + '.jpg'
#        img = load_img(img_path, target_size=(40, 40))
#        x = img_to_array(img)
#        movies_input_shifted_delayed.append(x)
#    movies_input_shifted.append(movies_input_shifted_delayed)


#print(np.array(movies_input).shape[0])
#print(np.array(movies_input).shape[1])
#print(np.array(movies_input).shape[2])
#print(np.array(movies_input).shape[3])
#print(np.array(movies_input).shape[4])

#print(np.array(movies_input_shifted).shape[0])
#print(np.array(movies_input_shifted).shape[1])
#print(np.array(movies_input_shifted).shape[2])
#print(np.array(movies_input_shifted).shape[3])
#print(np.array(movies_input_shifted).shape[4])

#--------------------------------------------------------------------------------------------------#

import os
rootdir = 'C:\\Users\\DanJas\\Desktop\\CNNLSTM'

movies_input = []
movies_input_shifted = []

for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        movies_input_delayed = []
        movies_input_shifted_delayed = []
        for files in os.walk(dir):
            for i in range(len(files[2])):
                img_path = 'C:\\Users\\DanJas\\Desktop\\CNNLSTM\\' + str(dir) + '\\' + str(files[2][i])
                img = load_img(img_path, target_size=(40,40))
                x = img_to_array(img)
                movies_input_delayed.append(x)
        movies_input.append(movies_input_delayed[:-1])
        movies_input_shifted.append(movies_input_delayed[1:])

print(np.array(movies_input).shape[0])
print(np.array(movies_input).shape[1])
print(np.array(movies_input).shape[2])
print(np.array(movies_input).shape[3])
print(np.array(movies_input).shape[4])

print(np.array(movies_input_shifted).shape[0])
print(np.array(movies_input_shifted).shape[1])
print(np.array(movies_input_shifted).shape[2])
print(np.array(movies_input_shifted).shape[3])
print(np.array(movies_input_shifted).shape[4])

#Train the network
### Was
#noisy_movies, shifted_movies = generate_movies(n_samples=120)
#seq.fit(noisy_movies[:100], shifted_movies[:100], batch_size=1,
#        epochs=10, validation_split=0.05)
### Now with own images is
seq.fit(np.array(movies_input), np.array(movies_input_shifted), batch_size=1,
        epochs=10)


# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 90
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))
