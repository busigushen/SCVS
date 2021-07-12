"""
@File         : scvs_input.py
@Time         : 2021/07/11
@Author       : Ji Jiayu
@Update       : 
@Discription  : convert original data to standard tensorflow data '.tfrecords';
                generate image and label batch.
"""


# from __future__ import unicode_literals
# from __future__ import print_function
# from __future__ import division
import h5py
import os
import gzip
import numpy
import PIL.Image as Image
import scipy.io as io
import pandas as pd

from tensorflow.python.platform import gfile
import tensorflow as tf

# dirs
DATA_DIR = './data'
DATA_DIR4 ='./data4'
DATA_DIR5 ='./data5'

LOG_DIR5 = './first_step_logs'
LOG_DIR7 = './second_step_logs'

# path of mnist

DIR = 'H:\\IQA'
DATABASE = 'tid2013'
MOS_PATH = DIR + '\\' + DATABASE + '\\mos_with_names.txt'
REF_PATH = DIR + '\\' + DATABASE + '\\reference_images\\'
DIS_PATH = DIR + '\\' + DATABASE + '\\distorted_images\\'

# information of mnist
HEIGHT = 512
WIDTH = 512
DEPTH = 3

PATCH_SIZE = 32
NUM_PATCHES_PER_IMAGE = 32

TRAIN_DATA_NUM = 18
VAL_DATA_NUM = 6
TEST_DATA_NUM = 6
NUM_PER_IMAGE = 5


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
        :param f: file object. a file object that can be passed into a gzip reader.

    Returns:
        :return data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
        :exception ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)

        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
        :param f: file object. A file object that can be passed into a gzip reader.
        :param one_hot: bool. Does one hot encoding for the result.
        :param num_classes: int. Number of classes for the one hot encoding.

    Returns:
        :returns labels: ndarray. a 1D uint8 numpy array.

    Raises:
        :exception ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)

        return labels


def load_img(path, gray_scale=False):
    """ Load image and convert to Image object.

    Args:
        :param path:str. the path of the image file.
        :param gray_scale:bool. gray or color.

    Return:
        :return img:Image object. an instance of Image object.
    """
    img = Image.open(path)
    if gray_scale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(y, z, filename):
    """Converts data to tfrecords.

    Args:
      :param x, y: list - [img1, img2, ...].
                    img: ndarray.
      :param name: str. 
    """
    if not gfile.Exists(filename):
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(len(z)):
            
            image_y = y[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _float_feature(z[index]),
                
                'image_y': _bytes_feature(image_y)
            }))
            writer.write(example.SerializeToString())
        writer.close()


def convert_to_jjy_5(y, z,vs, filename):
    """Converts data to tfrecords.

    Args:
      :param x, y: list - [img1, img2, ...].
                    img: ndarray.
      :param name: str. 
    """
    if not gfile.Exists(filename):
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(len(z)):
            print(y[index].shape)
            image_y = y[index].tostring()
            VS=vs[index].tostring()
            #image_y = y[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                
                'mos'  :_float_feature(z[index]),
                'vs'  :_bytes_feature(VS),
                'image_y': _bytes_feature(image_y)
                
            }))
            writer.write(example.SerializeToString())
        writer.close()


def convert_to_jjy_6(y, z, simmatrixc , pcm, filename):
    """Converts data to tfrecords.

    Args:
      :param x, y: list - [img1, img2, ...].
                    img: ndarray.
      :param name: str. 
    """
    if not gfile.Exists(filename):
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(len(z)):
            print(y[index].shape)
            image_y = y[index].tostring()
            
            SimMatrixC=simmatrixc[index].tostring()
            PCm=pcm[index].tostring()
            
            #image_y = y[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                
                'mos'  :_float_feature(z[index]),                
                'SimMatrixC': _bytes_feature(SimMatrixC), 
                'PCm':_bytes_feature(PCm),
                
                'image_y': _bytes_feature(image_y)
                
            }))
            writer.write(example.SerializeToString())
        writer.close()
        
        
def load_data():
    """Load database, convert to Examples and write the result to TFRecords."""
    # Load data
    
    df=pd.read_excel('csiq.DMOS.xlsx',sheet_name='all_by_image')
    excel=df.values[3:,3:]
    pic_name=excel[:,0]
    dis_level=excel[:,3]

    dis_type=excel[:,2]
    dmos=excel[:,5]

    dis_img_set = []
    mos_set = dmos
    for i in range(pic_name.shape[0]):
        if pic_name[i]==1600:
            pic_content='1600'
        else:
            pic_content=pic_name[i]
            
        if dis_type[i]=='noise':
            DIS_TYPE='AWGN'
        elif dis_type[i]=='jpeg':
            DIS_TYPE='JPEG'
        elif dis_type[i]=='blur':
            DIS_TYPE='BLUR'
            
        elif dis_type[i]=='jpeg 2000':
            DIS_TYPE='jpeg2000'
        else :
            DIS_TYPE=dis_type[i]
        
        DIS_LEVEL=str(dis_level[i])
            
        path='./dst_imgs\\'+pic_content+'.'+DIS_TYPE+'.'+DIS_LEVEL+'.png'
    
        dis_img_set.append(numpy.asarray(load_img(path, gray_scale=False), dtype=numpy.uint8))
        
    y=[]
    z=[]
    num=0
    # Convert to Examples and write the result to TFRecords.
    for i in range(pic_name.shape[0]):

        y.append(dis_img_set[i])
        z.append(mos_set[i])
        if i == 865:
            filename = os.path.join(DATA_DIR, 'image_' + str(num) + '.tfrecords')
            convert_to(y, z, filename)
        
        elif not dis_type[i]==dis_type[i+1]:
            if not gfile.Exists(os.path.join(DATA_DIR)):
                os.makedirs(os.path.join(DATA_DIR))

            filename = os.path.join(DATA_DIR, 'image_' + str(num) + '.tfrecords')
            convert_to(y, z, filename)
            num += 1
            y = []
            z = []


def jjy_load_data_5():
    """Load database, convert to Examples and write the result to TFRecords."""
    # Load data
    df = pd.read_excel('csiq.DMOS.xlsx',sheet_name='all_by_image')
    excel = df.values[3:,3:]
    pic_name = excel[:,0]
    dis_level = excel[:,3]

    dis_type = excel[:,2]
    dmos = excel[:,5]

    dis_img_set = []
    mos_set = dmos
    for i in range(pic_name.shape[0]):
        if pic_name[i] == 1600:
            pic_content = '1600'
        else:
            pic_content = pic_name[i]
            
        if dis_type[i] == 'noise':
            DIS_TYPE = 'AWGN'
        elif dis_type[i] == 'jpeg':
            DIS_TYPE = 'JPEG'
        elif dis_type[i] == 'blur':
            DIS_TYPE = 'BLUR'
            
        elif dis_type[i] == 'jpeg 2000':
            DIS_TYPE = 'jpeg2000'
        else :
            DIS_TYPE =dis_type[i]
        
        DIS_LEVEL=str(dis_level[i])
            
        path='./dst_imgs\\'+pic_content+'.'+DIS_TYPE+'.'+DIS_LEVEL+'.png'
    
        dis_img_set.append(numpy.asarray(load_img(path, gray_scale=False), dtype=numpy.uint8))

    VS_MAP=h5py.File('./VSMap_full_size.mat')['VSMap_MAT']
    VS_MAP=numpy.transpose(VS_MAP) #要转置
    
    y = []
    z = []
    vs_map = []

    num = 0
    # Convert to Examples and write the result to TFRecords.
    for i in range(pic_name.shape[0]):

        y.append(dis_img_set[i])
        z.append(mos_set[i])
        vs_map.append(VS_MAP[i])
        
        if i == 865:
            filename = os.path.join(DATA_DIR4, 'image_' + str(num) + '.tfrecords')
            convert_to_jjy_5(y,z,vs_map, filename)

        elif not dis_type[i] == dis_type[i+1]:
            if not gfile.Exists(os.path.join(DATA_DIR4)):
                os.makedirs(os.path.join(DATA_DIR4))

            filename = os.path.join(DATA_DIR4, 'image_' + str(num) + '.tfrecords')
            convert_to_jjy_5(y,z,vs_map, filename)

            num+=1
            y=[]
            z=[]
            vs_map=[]

        
def jjy_load_data_6():
    """Load database, convert to Examples and write the result to TFRecords."""

    # Load data
    
    df=pd.read_excel('csiq.DMOS.xlsx',sheet_name='all_by_image')
    excel=df.values[3:,3:]
    pic_name=excel[:,0]
    dis_level=excel[:,3]

    dis_type=excel[:,2]
    dmos=excel[:,5]

    dis_img_set = []
    mos_set = dmos
    for i in range(pic_name.shape[0]):
        if pic_name[i]==1600:
            pic_content='1600'
        else:
            pic_content=pic_name[i]
            
        if dis_type[i]=='noise':
            DIS_TYPE='AWGN'
        elif dis_type[i]=='jpeg':
            DIS_TYPE='JPEG'
        elif dis_type[i]=='blur':
            DIS_TYPE='BLUR'
            
        elif dis_type[i]=='jpeg 2000':
            DIS_TYPE='jpeg2000'
        else :
            DIS_TYPE=dis_type[i]
        
        DIS_LEVEL=str(dis_level[i])
            
        path='./dst_imgs\\'+pic_content+'.'+DIS_TYPE+'.'+DIS_LEVEL+'.png'
    
        dis_img_set.append(numpy.asarray(load_img(path, gray_scale=False), dtype=numpy.uint8))
        
    SimMatrixC=h5py.File('./SimMatrixC.mat')['SimMatrixC']
    SimMatrixC=numpy.transpose(SimMatrixC) #要转置
    
    PCm=h5py.File('./PCm.mat')['PCm']
    PCm=numpy.transpose(PCm) #要转置
        
    y=[]
    z=[]
    simmatrixc=[]
    pcm=[]
    num=0
    # Convert to Examples and write the result to TFRecords.
    for i in range(pic_name.shape[0]):
        y.append(dis_img_set[i])
        z.append(mos_set[i])
        simmatrixc.append(SimMatrixC[i])
        pcm.append(PCm[i])
        if i==865:
            filename = os.path.join(DATA_DIR5, 'image_' + str(num) + '.tfrecords')
            convert_to_jjy_6(y,z,simmatrixc , pcm, filename)
        elif not dis_type[i]==dis_type[i+1]:
            if not gfile.Exists(os.path.join(DATA_DIR5)):
                os.makedirs(os.path.join(DATA_DIR5))

            filename = os.path.join(DATA_DIR5, 'image_' + str(num) + '.tfrecords')
            convert_to_jjy_6(y,z,simmatrixc, pcm, filename)
            num+=1
            y=[]
            z=[]
            simmatrixc=[]
            pcm=[]
            


def read_and_decode(filename_queue):
    """Reads and parses examples from data files .tfrecords.

    Args:
        :param filename_queue: queue. A queue of strings with the filenames to read from. 

    Returns:
        :return result: DataRecord. An object representing a single example, with the following fields:
            label: an int32 Tensor.
            image_x, image_y: a [height*width*depth] uint8 Tensor with the image data.
    """

    class DataRecord(object):
        pass

    result = DataRecord()
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature([], tf.float32),
            
            'image_y': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor to a uint8 tensor
    
    result.image_y = tf.decode_raw(features['image_y'], tf.uint8)

    result.image_y.set_shape([HEIGHT*WIDTH*DEPTH])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    result.label = features['label']

    return result


def read_and_decode_jjy_4(filename_queue):
    """Reads and parses examples from data files .tfrecords.

    Args:
        :param filename_queue: queue. A queue of strings with the filenames to read from. 

    Returns:
        :return result: DataRecord. An object representing a single example, with the following fields:
            label: an int32 Tensor.
            image_x, image_y: a [height*width*depth] uint8 Tensor with the image data.
    """

    class DataRecord(object):
        pass
    print(filename_queue)
    result = DataRecord()
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            
            'mos':tf.FixedLenFeature([], tf.float32),
            'vs':tf.FixedLenFeature([], tf.string),
            #'image_x': tf.FixedLenFeature([], tf.string),
            'image_y': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor to a uint8 tensor
    result.image_y = tf.decode_raw(features['image_y'], tf.uint8)
    result.image_y.set_shape([16*16*32*32*DEPTH])
    result.vs=tf.decode_raw(features['vs'],tf.float64)
    result.vs.set_shape([512*512])
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    result.mos=features['mos']
    print(result.image_y.shape)
    print(result.image_y)

    return result


def read_and_decode_jjy_5(filename_queue):
    """Reads and parses examples from data files .tfrecords.

    Args:
        :param filename_queue: queue. A queue of strings with the filenames to read from. 

    Returns:
        :return result: DataRecord. An object representing a single example, with the following fields:
            label: an int32 Tensor.
            image_x, image_y: a [height*width*depth] uint8 Tensor with the image data.
    """

    class DataRecord(object):
        pass

    
    print(filename_queue)
    result = DataRecord()
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            
            'mos':tf.FixedLenFeature([], tf.float32),
            'SimMatrixC':tf.FixedLenFeature([], tf.string),
            
            'PCm': tf.FixedLenFeature([], tf.string),
            'image_y': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor to a uint8 tensor
    #result.image_x = tf.decode_raw(features['image_x'], tf.uint8)
    result.image_y = tf.decode_raw(features['image_y'], tf.uint8)
    #result.image_x.set_shape([HEIGHT*WIDTH*DEPTH])
    result.image_y.set_shape([16*16*32*32*DEPTH])
    
    
    result.simmatrixc=tf.decode_raw(features['SimMatrixC'],tf.float64)
    result.simmatrixc.set_shape([512*512])
    
    result.pcm=tf.decode_raw(features['PCm'],tf.float64)
    result.pcm.set_shape([512*512])
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    
    result.mos=features['mos']
    
    
    print(result.image_y.shape)
    print(result.image_y)

    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        :param image: tuple - (image_x, image_y). 
                imagex, image_y: 3-D Tensor of [height, width, 3] of type.float32.
        :param label: 1-D Tensor of type.float32
        :param min_queue_examples: int32, minimum number of samples to retain 
        in the queue that provides of batches of examples.
        :param batch_size: Number of images per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        :return images_x, images_y: Images. 4D tensor of [batch_size, height, width, 3] size.
        :return labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        images_y, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images_y, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.

    tf.summary.image('batch_images_y', tensor=images_y, max_outputs=4)

    return images_y, label_batch


def _generate_image_and_label_batch_jjy_4(image, mos,vs, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        :param image: tuple - (image_x, image_y). 
                imagex, image_y: 3-D Tensor of [height, width, 3] of type.float32.
        :param label: 1-D Tensor of type.float32
        :param min_queue_examples: int32, minimum number of samples to retain 
        in the queue that provides of batches of examples.
        :param batch_size: Number of images per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        :return images_x, images_y: Images. 4D tensor of [batch_size, height, width, 3] size.
        :return labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        
        images_y, mos_batch,vs_batch = tf.train.shuffle_batch(
            [image, mos,vs],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        
        images_y, label_batch = tf.train.batch(
            [image[0],  label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.

    tf.summary.image('batch_images_y', tensor=images_y, max_outputs=4)
    print(images_y.shape)

    return images_y, mos_batch,vs_batch

def _generate_image_and_label_batch_jjy_5(image, mos, simmatrixc, pcm, min_queue_examples,        # here  2019.08.27
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        :param image: tuple - (image_x, image_y). 
                imagex, image_y: 3-D Tensor of [height, width, 3] of type.float32.
        :param label: 1-D Tensor of type.float32
        :param min_queue_examples: int32, minimum number of samples to retain 
        in the queue that provides of batches of examples.
        :param batch_size: Number of images per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        :return images_x, images_y: Images. 4D tensor of [batch_size, height, width, 3] size.
        :return labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        
        images_y, mos_batch, simmatrixc_batch, pcm_batch = tf.train.shuffle_batch(
            [image, mos,simmatrixc, pcm],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        
        images_y, label_batch = tf.train.batch(
            [image[0],  label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.

    tf.summary.image('batch_images_y', tensor=images_y, max_outputs=4)
    print(images_y.shape)

    return images_y, mos_batch, simmatrixc_batch, pcm_batch

def random_sample( images_y, patch_size, num_patches):
    """random sample patch pairs from image pairs.
    
    Args:
        :param images_x, images_y: tensor - (batch_size, height, width, depth). 
        :param patch_size: int. 
        :param num_patches: int. we crop num_patches patches from each image pair.
        
    Returns:
        :return patches_x, patches_y: tensor - (batch_size*num_patches, patch_size, patch_size, depth). 
    """
    
    patches_y = []

    
    for i in range(images_y.get_shape()[0].value):
        for j in range(num_patches):
            # Randomly crop a [height, width] section of the image.
            patch_y = tf.random_crop(images_y[i, :, :, :], [patch_size, patch_size, DEPTH])
            patches_y.append(patch_y)

    patches_y = tf.convert_to_tensor(value=patches_y, dtype=tf.float32, name='sampled_patches_y')

    return patches_y


def random_sample_jjy_4(images_y, vs, num_patches):
    """random sample patch pairs from image pairs.
    
    Args:
        :param images_x, images_y: tensor - (batch_size, height, width, depth). 
        :param patch_size: int. 
        :param num_patches: int. we crop num_patches patches from each image pair.
        
    Returns:
        :return patches_x, patches_y: tensor - (batch_size*num_patches, patch_size, patch_size, depth). 
    """
    patches_y = []
    patches_vs=[]

    images_yvs = tf.concat([images_y, vs], axis=3)
    for i in range(images_y.get_shape()[0].value):
        for j in range(num_patches):
            # Randomly crop a [height, width] section of the image.

            
            patch_yvs = tf.random_crop(images_yvs[i, :, :, :], [96, 96, 4])
            patch_y=patch_yvs[:,:,:3]
            
            patch_vs=patch_yvs[32:64,32:64,3]
            patch_vs=tf.reduce_mean(patch_vs)
            
            for z in range(3):
                for n in range(3):
                    patches_y.append(patch_y[32*z:32*(z+1),32*n:32*(n+1),: ])
                        

            patches_vs.append(patch_vs)

    patches_vs = tf.convert_to_tensor(value=patches_vs, dtype=tf.float32, name='sampled_patches_vs')
    patches_y = tf.convert_to_tensor(value=patches_y, dtype=tf.float32, name='sampled_patches_y')
    patches_y = tf.reshape(patches_y, [-1,32,32,3])
    patches_vs = tf.reshape(patches_vs, [-1])

    return patches_y, patches_vs


def random_sample_jjy_5(images_y, mos_batch, simmatrixc_batch, pcm_batch, num_patches):
    """random sample patch pairs from image pairs.
    
    Args:
        :param images_x, images_y: tensor - (batch_size, height, width, depth). 
        :param patch_size: int. 
        :param num_patches: int. we crop num_patches patches from each image pair.
        
    Returns:
        :return patches_x, patches_y: tensor - (batch_size*num_patches, patch_size, patch_size, depth). 
    """
    #patches_x = []
    patches_y = []
    patches_labels=[]
    

    images_fus = tf.concat([images_y, simmatrixc_batch, pcm_batch], axis=3)
    for i in range(images_y.get_shape()[0].value):
        for j in range(num_patches):
            # Randomly crop a [height, width] section of the image.

            patch_fus = tf.random_crop(images_fus[i, :, :, :], [32, 32, 5])
            patch_y=patch_fus[:,:,:3]
            patches_y.append(patch_y)
            
            patch_simmatrixc=patch_fus[:,:,3]
            patch_simmatrixc=tf.reduce_sum(patch_simmatrixc)
            
            patch_pcm=patch_fus[:,:,4]
            patch_pcm=tf.reduce_sum(patch_pcm)
            
            patch_fsim=tf.divide(patch_simmatrixc,patch_pcm)
            
            patch_fsim = tf.cond(tf.is_nan(patch_fsim),lambda:tf.constant(0.5),lambda:patch_fsim)
            patch_fsim = tf.cond(tf.is_inf(patch_fsim),lambda:tf.constant(0.5),lambda:patch_fsim)
            
            patch_label = 0.9*mos_batch[i]+0.6*patch_fsim
            patches_labels.append(patch_label)

    patches_labels = tf.convert_to_tensor(value=patches_labels, dtype=tf.float32, name='sampled_patches_labels')
    patches_y = tf.convert_to_tensor(value=patches_y, dtype=tf.float32, name='sampled_patches_y')
    patches_y = tf.reshape(patches_y, [-1, 32, 32, 3])
    patches_labels = tf.reshape(patches_labels, [-1])

    return patches_y,patches_labels


def distorted_inputs_jjy_5(filenames, batch_size):
    """Construct distorted input for training using the Reader ops.

    Args:
        :param filenames: list - [str1, str2, ...].
        :param batch_size: int. 

    Returns:
       :returns: tuple - (patches_x, patches_y, labels).
                patches_x, patches_y: tensors - (batch_size*num_patches, patch_size, patch_size, depth). 
                lables: tensors - (batch_size).
    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.variable_scope('input'):
        # Create a queue that produces the filenames to read.
        
        print(filenames)
        filename_queue = tf.train.string_input_producer(string_tensor=filenames)

        # Even when reading in multiple threads, share the filename queue.
        result = read_and_decode_jjy_5(filename_queue)

        # OPTIONAL: Could reshape into a image and apply distortionshere.

        reshaped_image_y = tf.reshape(result.image_y, [512,512, DEPTH])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        distorted_image_y = tf.cast(reshaped_image_y, tf.float32) * (1. / 255) - 0.5

        mos = result.mos
        simmatrixc = tf.reshape(tf.cast(result.simmatrixc, dtype=tf.float32), [512, 512, 1])
        pcm = tf.reshape(tf.cast(result.pcm, dtype=tf.float32), [512, 512, 1])

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 1000
        print('Filling queue with %d mnist images before starting to train or validation. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        
        images_y, mos_batch, simmatrixc_batch, pcm_batch = \
            _generate_image_and_label_batch_jjy_5(image=distorted_image_y, mos=mos, simmatrixc=simmatrixc, pcm=pcm,
                                               min_queue_examples=min_queue_examples,
                                               batch_size=batch_size,
                                               shuffle=True)

        print(images_y, mos_batch,simmatrixc_batch, pcm_batch)

        # Random crop patches from images
        patches_y, patches_labels = random_sample_jjy_5(images_y, mos_batch, simmatrixc_batch, pcm_batch, PATCH_SIZE)
        

        # Display the training images in the visualizer.
        tf.summary.image('patches_y', tensor=patches_y, max_outputs=4)

        return patches_y, mos, patches_labels


def distorted_inputs_jjy_6(filenames, batch_size):
    """Construct distorted input for training using the Reader ops.

    Args:
        :param filenames: list - [str1, str2, ...].
        :param batch_size: int. 

    Returns:
       :returns: tuple - (patches_x, patches_y, labels).
                patches_x, patches_y: tensors - (batch_size*num_patches, patch_size, patch_size, depth). 
                lables: tensors - (batch_size).
    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.variable_scope('input'):
        # Create a queue that produces the filenames to read.
        
        print(filenames)
        filename_queue = tf.train.string_input_producer(string_tensor=filenames)

        # Even when reading in multiple threads, share the filename queue.
        result = read_and_decode_jjy_4(filename_queue)
        
        

        # OPTIONAL: Could reshape into a image and apply distortionshere.

        reshaped_image_y = tf.reshape(result.image_y, [512,512, DEPTH])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        #distorted_image_x = tf.cast(reshaped_image_x, tf.float32) * (1. / 255) - 0.5
        distorted_image_y = tf.cast(reshaped_image_y, tf.float32) * (1. / 255) - 0.5

        mos=result.mos
        vs=tf.reshape(tf.cast(result.vs,dtype=tf.float32),[512,512,1])

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 1000
        print('Filling queue with %d mnist images before starting to train or validation. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        images_y, moses ,vses = \
            _generate_image_and_label_batch_jjy_4(image=distorted_image_y, mos=mos,vs=vs,
                                               min_queue_examples=min_queue_examples,
                                               batch_size=batch_size,
                                               shuffle=True)
       
        print(images_y, moses, vses)

        # Random crop patches from images

        patches_y, patches_vs = random_sample_jjy_4( images_y, vses, 18)
        

        # Display the training images in the visualizer.

        tf.summary.image('patches_y', tensor=patches_y, max_outputs=4)

        return patches_y, moses, patches_vs

    
def inputs(filenames, batch_size):
    """Construct input without distortion for MNIST using the Reader ops.

    Args:
        :param filenames: list - [str1, str2, ...].
        :param batch_size: int. 

    Returns:
       :returns: tuple - (images, labels).
                images: tensors - [batch_size, height*width*depth].
                lables: tensors - [batch_size].
    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.variable_scope('input_evaluation'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(string_tensor=filenames)

        # Even when reading in multiple threads, share the filename
        # queue.
        result = read_and_decode(filename_queue)

        # OPTIONAL: Could reshape into a image and apply distortionshere.
        #reshaped_image_x = tf.reshape(result.image_x, [HEIGHT, WIDTH, DEPTH])
        reshaped_image_y = tf.reshape(result.image_y, [HEIGHT, WIDTH, DEPTH])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image_y = tf.cast(reshaped_image_y, tf.float32) * (1. / 255) - 0.5
        label = result.label

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 500
        print('Filling queue with %d mnist images before starting to train or validation. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        images_y, labels = \
            _generate_image_and_label_batch(image=image_y, label=label,
                                               min_queue_examples=min_queue_examples,
                                               batch_size=batch_size,
                                               shuffle=True)

        # Random crop patches from images
        patches_y = random_sample( images_y, PATCH_SIZE, NUM_PATCHES_PER_IMAGE)

        return patches_y, labels