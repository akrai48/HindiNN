import os
import zipfile

import tensorflow as tf


def hindi_all(batch_size=50, min_after_dequeue=100):
    dataset_file_path = '../../dataset'
    if not (os.path.exists(os.path.join(dataset_file_path,'train-all-jpeg') or
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

    # Apparently tensorflow needs to add some fixed-sized metadata to tensor
    # to be able to use batch function on it
    image.set_shape([28, 28, 1])

    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
          [image, label], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)

    # Start a new session to show example output.
    with tf.Session() as sess:
        # Required to get the filename matching to run.
        tf.initialize_all_variables().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Run session to get batches of image and label.
        train_image_batch, train_label_batch = sess.run([image_batch, label_batch])

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

    return train_image_batch, train_label_batch
