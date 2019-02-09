import boto3
import os
import os.path
import tensorflow as tf
import cv2
import numpy as np

service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
bucket_name = 'train-data'

# data read하는 위치 : 현재위치의  TrainData folder
DATASET_PATH = os.getcwd() + '/TrainData'

MODE = 'folder'

IMG_HEIGHT = 100
IMG_WIDTH = 100
CHANNELS = 3

N_CLASSES = 1

save_path = os.getcwd()
model_name = 'tf-trainmodel'

# 모델 save : 현재 폴더에 생성
save_path_full = os.path.join(save_path, model_name)


s3 = boto3.client(service_name, endpoint_url=endpoint_url,
                  aws_access_key_id=os.environ['NCLOUD_ACCESS_KEY'], aws_secret_key_id=os.environ['NCLOUD_SECRET_KEY'])

response = s3.list_objects(
    Bucket=bucket_name, Delimiter='/images', MaxKeys=300)

images = []

while True:
    print('IsTruncated=%r' % response.get('IsTruncated'))
    print('Marker=%s' % response.get('Marker'))
    print('NextMarker=%s' % response.get('NextMarker'))

    print('Object List')
    for content in response.get('Contents'):
        print(' Name=%s, Size=%d, Owner=%s' %
              (content.get('Key'), content.get('Size'), content.get('Owner').get('ID')))
        images.append(content.get('Key'))

    if response.get('IsTruncated'):
        response = s3.list_objects(Bucket=bucket_name, Delimiter='/images', MaxKeys=300,
                                   Marker=response.get('NextMarker'))
    else:
        break

if not os.path.isdir('TrainData'):
    os.mkdir('TrainData')

for image in images:
    s3.download_file(bucket_name, '/images/' + image,
                     os.getcwd() + '/TrainData/' + image)

response = s3.list_objects(
    Bucket=bucket_name, Delimiter='/models', MaxKeys=300)

contents = response.get('Contents')

if len(contents) > 0 and 'tf-trainmodel' in [x.get('Key') for x in contents]:
    model_exists = True
    s3.download_file(bucket_name, '/models/tf-trainmodel',
                     save_path_full)
else:
    model_exists = False


def read_image(path):
    path = str(path, 'utf-8')
    return cv2.imread(path, cv2.IMREAD_COLOR)


def read_data(path, label):
    image = read_image(path)
    label = np.array(label, dtype=np.int32)
    image = image.astype(np.int32)
    return image, label


def data_resize_fuction(image_decoded, label):
    image_decoded.set_shape([None, None, CHANNELS])
    image_resized = tf.image.resize_images(
        image_decoded, [IMG_HEIGHT, IMG_WIDTH])
    return image_resized, label


def data_normalization(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


def input_data(dataset_path, mode, batch_size):
    imagepaths, labels, classes = list(), list(), list()
    label_index_dict, label_name_dict = {}, {}
    label_index = 0

    if mode == 'folder':
        files = os.listdir(DATASET_PATH)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for file in files:
            label = file.split('_')[0]
            if not label in classes:
                classes.append(label)
                label_index_dict[label_index] = label
                label_name_dict[label] = label_index
                label_index += 1
            if file.endswith('.jpg'):
                imagepaths.append(os.path.join(DATASET_PATH, file))
                labels.append(label_name_dict[label])

        global N_CLASSES
        N_CLASSES = len(classes)
        print(classes)

    else:
        raise Exception("Unknown mode")

    dataset = tf.data.Dataset.from_tensor_slices((imagepaths, labels))
    dataset = dataset.map(lambda images, labels:
                          tuple(tf.py_func(read_data, [images, labels], [tf.int32, tf.int32])))

    dataset = dataset.map(data_resize_fuction)
    dataset = dataset.map(data_normalization)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=(
        int(len(imagepaths) * 0.4) + 3 * batch_size))
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()
    image_stacked, label_stacked = iterator.get_next()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        image, label = sess.run([image_stacked, label_stacked])

    return dataset


learning_rate = 0.001
num_steps = 100
batch_size = 128
display_step = 10

dropout = 0.75


dataset = input_data(DATASET_PATH, MODE, batch_size)


def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        # change value later
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # fully connected
        affine = tf.contrib.layers.flatten(conv2)
        affine = tf.layers.dense(affine, 1024)
        affine = tf.layers.dropout(affine, rate=dropout, training=is_training)

        out = tf.layers.dense(affine, n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out


iterator = dataset.make_initializable_iterator()
X, Y = iterator.get_next()

print("N_CLASSES")
print(N_CLASSES)
# false로 했을때 value error
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    if model_exists:
        saver.restore(sess, save_path_full)
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)

    for step in range(1, num_steps+1):
        if step % display_step == 0:
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))
        else:
            sess.run(train_op)

    print("Optimization Finished!")
    saver.save(sess, save_path_full)

for image in images:
    s3.delete_file(Bucket=bucket_name, Key='/images/' + image)

s3.upload_file(save_path_full, bucket_name, model_name)
