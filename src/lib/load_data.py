import os

import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt

dataset_file_path = '../../dataset'
if not ( os.path.exists(os.path.join(dataset_file_path,'train-all-jpeg') or
                           os.path.exists(os.path.join(dataset_file_path,'test-all-jpeg')))):
    data_zip_file = 'hindi-all-jpeg-28x28.zip'
    file_ref = zipfile.ZipFile(os.path.join(dataset_file_path, data_zip_file), 'r')
    file_ref.extractall(dataset_file_path)
    file_ref.close()

train_filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once(os.path.join(dataset_file_path,
                                               'train-all-jpeg',
                                               '*jpeg')))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
label, image_file = image_reader.read(train_filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)
#label_name = tf.string_split(image, '\\')

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    images=[]
    labels=[]
    for i in range(18357):
        label_tensor, image_tensor = sess.run([label, image])
        drive, label_path_and_file = os.path.splitdrive(str(label_tensor))
        path, label_name = os.path.split(label_path_and_file)
        images.append(image_tensor)
        labels.append(label_name)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

image_set, label_set = tf.constant(images), tf.constant(labels)