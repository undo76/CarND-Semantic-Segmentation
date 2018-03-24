import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = graph.get_tensor_by_name(
        vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = graph.get_tensor_by_name(
        vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(
        vgg_layer7_out_tensor_name)

    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor


tests.test_load_vgg(load_vgg, tf)


def conv1x1(input, filters):
    """
    Create a convolutional 1x1 layer.
    :param input: TF Tensor
    :param filters: Number of output filters
    :return: A Tensor
    """

    # Add non-linear activation function
    l2 = tf.contrib.layers.l2_regularizer(.001)
    return tf.layers.conv2d(
        input, filters, 1, padding='same', kernel_regularizer=l2)


def upsample(input, filters, kernel_size, strides):
    """
    Upsample a layer using a transposed convolution.
    :param input: TF tensor
    :param filters: Number of output filters
    :param strides: Strides of the convolution
    :return: A Tensor
    """

    # TODO: Add non-linear activation function
    l2 = tf.contrib.layers.l2_regularizer(.001)
    return tf.layers.conv2d_transpose(
        input, filters, kernel_size, strides=strides, padding='same', kernel_regularizer=l2)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # To investigate: Use concat instead of add classes.
    # Also we are not adding dropout (It is used for VGG layers)

    # Add a 1x1 convolution to reduce dimensionality to the output layers -> num_classes
    pred_3 = conv1x1(vgg_layer3_out, num_classes)
    pred_4 = conv1x1(vgg_layer4_out, num_classes)
    pred_7 = conv1x1(vgg_layer7_out, num_classes)

    # Upsample last layer and combine (x2)
    pred_7x2 = upsample(pred_7, num_classes, 4, strides=(2, 2))
    pred_4_7x2 = tf.add(pred_4, pred_7x2)

    # Upsample and combine (x2)
    pred_4x2_7x4 = upsample(pred_4_7x2, num_classes, 4, strides=(2, 2))
    pred_3_4x2_7x4 = tf.add(pred_3, pred_4x2_7x4)

    # Upsample to final size (x8)
    pred_3x8_4x16_7x32 = upsample(
        pred_3_4x2_7x4, num_classes, 16, strides=(8, 8))

    return pred_3x8_4x16_7x32


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # We don't need to reshape.
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    # logits = nn_last_layer
    # labels = correct_label

    # Calculate cross_entropy with logits
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels))

    # Add L2 regularization term to the loss
    lambda_l2 = 1   # L2 regularization coheficient
    loss = cross_entropy_loss + lambda_l2 * tf.losses.get_regularization_loss()

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    for epoch_index in range(epochs):
        current_epoch_loss = 0

        print("Running epoch: " + str(epoch_index + 1))
        for X, y in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={ input_image:X, correct_label:y, keep_prob:0.8 })
            current_epoch_loss += loss
        
        print("Training loss for epoch: " + str(current_epoch_loss))
    pass


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    learning_rate = 0.01
    epochs = 1
    batch_size = 20
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(
            tf.float32, [None, image_shape[0], image_shape[1], num_classes])

        # Build NN using load_vgg, layers, and optimize function
        vgg_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        nn = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, vgg_input, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
