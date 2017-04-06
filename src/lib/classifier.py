import keras


def create_classifier_graph(**kwargs):
    classifier_type = kwargs['classifier_type']
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(kernel_size=3, input_shape=(28, 28, 1)))
    model.add(keras.layers.Dense())
