from import_data import create_spectrogram
from slice_spectogram import slice_spect
from load_data import load_dataset
import numpy as np
from keras.models import Model, load_model


loaded_model = load_model("Saved_Model/Model.h5")
loaded_model.set_weights(loaded_model.get_weights())

matrix_size = loaded_model.layers[-2].output.shape[1]
new_model = Model(loaded_model.inputs, loaded_model.layers[-2].output)

images, labels = load_dataset(verbose=1, mode="Test")
images = np.expand_dims(images, axis=3)

images = images / 255.

ynew = np.argmax(loaded_model.predict(images), axis=-1)
for i in range(len(images)):
    print("X=%s, Predicted=%s" % (i, ynew[i]))