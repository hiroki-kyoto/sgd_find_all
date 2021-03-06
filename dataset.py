# dataset.py
# Provide functions to load dataset of CIFAR-100,
import pickle
import numpy as np
from ctypes import *
import time
from PIL import Image as pi
import matplotlib.pyplot as plt
import os
import struct

class Dataset(object):
  '''
    Dataset: unifying the dataset APIs with loading, 
    and feeding samples into models.
  '''
  def __init__(self):
    self.RAND_SEQ_LENGTH = 1024*1024
    self.initialized = False
    pass
  
  def ready(self):
    return self.initialized
  
  '''
    ds_open_fn : dataset open function
    ds_path : dataset path
    function : get samples and multi-labels
  '''
  def load(self, ds_open_fn, ds_path):
    (self.train_samples, 
    self.train_labels, 
    self.test_samples,
    self.test_labels) = ds_open_fn(ds_path)
    # prepare a random sequence for generating training batches
    self.rand_seq = np.random.randint(0, len(self.train_labels), self.RAND_SEQ_LENGTH)
    self.seq_id = 0
    self.batch_size = 0
    self.initialized = True
  
  ''' set the batch size '''
  def set_batch_size(self, size):
    assert(size <= self.RAND_SEQ_LENGTH)
    assert(self.initialized)
    self.batch_size = size
    self.sample_batch = np.zeros([
      size, 
      self.train_samples.shape[1], 
      self.train_samples.shape[2],
      self.train_samples.shape[3]])
    self.label_batch = np.zeros([size])
  
  ''' generate a batch with given size for training '''
  def train_batch(self):
    assert(self.initialized)
    assert(self.batch_size)
    if self.seq_id + self.batch_size > self.RAND_SEQ_LENGTH:
      # regenerate the random sequence
      self.rand_seq = np.random.randint(0, len(self.train_labels), self.RAND_SEQ_LENGTH)
      self.seq_id = 0
    id_beg = self.seq_id
    id_end = self.seq_id + self.batch_size
    self.sample_batch = self.train_samples[self.rand_seq[id_beg : id_end]]
    self.label_batch = self.train_labels[self.rand_seq[id_beg : id_end]]
    self.seq_id += self.batch_size
    return (self.sample_batch, self.label_batch)
  
  ''' generate all sample label pairs for test '''
  def test(self):
    assert(self.initialized)
    return (self.test_samples, self.test_labels)

  def shape_of_sample(self):
    assert(self.initialized)
    return self.test_samples[0].shape

  def shape_of_label(self):
    assert(self.initialized)
    return np.max(self.test_labels)+1

def CIFAR100(path):
  file_train = path + '/train'
  file_test = path + '/test'
  tools = cdll.LoadLibrary('./libdatasettools.so')
  
  with open(file_train, 'rb') as fo:
    dict = pickle.load(fo)
    n, d = dict[b'data'].shape
    h = 32
    w = 32
    c = 3
    assert(d==h*w*c)
    train_samples = np.zeros([n, h, w, c], dtype=np.float32)
    train_labels = np.zeros([n], dtype=np.float32)
    data = dict[b'data'];
    
    if not train_samples.flags['C_CONTIGUOUS']:
      train_samples = np.ascontiguous(train_samples, dtype=train_samples.dtype)
    samples_ptr = cast(train_samples.ctypes.data, POINTER(c_float))
    
    if not data.flags['C_CONTIGUOUS']:
      data = np.ascontiguous(data, dtype=data.dtype)
    data_ptr = cast(data.ctypes.data, POINTER(c_uint8))
    
    train_labels = np.array(dict[b'fine_labels'])

    tools.NCHW2NHWC(samples_ptr, data_ptr, c_int(n), c_int(h), c_int(w), c_int(c))
    
  with open(file_test, 'rb') as fo:
    dict = pickle.load(fo)
    n, d = dict[b'data'].shape
    h = 32
    w = 32
    c = 3
    assert(d==h*w*c)
    test_samples = np.zeros([n, h, w, c], dtype=np.float32)
    test_labels = np.zeros([n], dtype=np.float32)
    data = dict[b'data'];
    
    if not test_samples.flags['C_CONTIGUOUS']:
      test_samples = np.ascontiguous(test_samples, dtype=test_samples.dtype)
    samples_ptr = cast(test_samples.ctypes.data, POINTER(c_float))
    
    if not data.flags['C_CONTIGUOUS']:
      data = np.ascontiguous(data, dtype=data.dtype)
    data_ptr = cast(data.ctypes.data, POINTER(c_uint8))
    
    test_labels = np.array(dict[b'fine_labels'])

    tools.NCHW2NHWC(samples_ptr, data_ptr, c_int(n), c_int(h), c_int(w), c_int(c))

  return (train_samples, 
          train_labels, 
          test_samples,
          test_labels)

def CIFAR10(path):
  num_batches = 5
  file_train = [path + '/data_batch_'+str(i) 
          for i in range(1,num_batches+1,1)]
  file_test = path + '/test_batch'
  tools = cdll.LoadLibrary('./libdatasettools.so')
  
  train_samples = []
  train_labels = []

  for i in range(5):
      with open(file_train[i], 'rb') as fo:
        dict = pickle.load(fo)
        n, d = dict[b'data'].shape
        train_samples.append(dict[b'data'].reshape(n*d))
        # labels
        train_labels.append(np.array(dict[b'labels']))
  
  train_samples = np.stack(train_samples, axis=0)
  train_samples = train_samples.reshape(50000,3,32,32)
  train_samples = train_samples.transpose([0,2,3,1])/255.0
  train_labels = np.stack(train_labels, axis=0)
  train_labels = train_labels.reshape([50000])
  
  with open(file_test, 'rb') as fo:
    dict = pickle.load(fo)
    n, d = dict[b'data'].shape
    test_samples = dict[b'data'].reshape([10000, 3, 32, 32])
    test_samples = test_samples.transpose([0,2,3,1])/255.0
    test_labels = np.array(dict[b'labels'])
    
  return (train_samples, 
          train_labels, 
          test_samples,
          test_labels)


def load_mnist_images(path):
    bin_file = open(path, 'rb')
    buf = bin_file.read()
    index = 0 
    magic, num_images, num_rows, num_cols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')
    if magic!=2051:
        raise NameError('MNIST dataset has bad format!')
    ims = np.zeros([num_images, num_rows*num_cols])
    for i in range(num_images):
        ims[i, :] = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
    ims = ims.reshape([num_images, num_rows, num_cols,1])/255.0
    return ims

def load_mnist_labels(path):
    bin_file = open(path, 'rb')
    buf = bin_file.read()
    index = 0
    magic, num_labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    if magic!=2049:
        raise NameError('MNIST dataset has bad format!')
    lbs = np.zeros([num_labels])
    lbs[:] = struct.unpack_from('>'+str(num_labels)+'B', buf, index)
    return lbs

def MNIST(path):
    # reading MINST database
    fn_train_images = 'train-images-idx3-ubyte'
    fn_train_labels = 'train-labels-idx1-ubyte'
    fn_test_images = 't10k-images-idx3-ubyte'
    fn_test_labels = 't10k-labels-idx1-ubyte'
    # load training images
    fn = os.path.join(path, fn_train_images)
    train_images = load_mnist_images(fn)
    # load training labels
    fn = os.path.join(path, fn_train_labels)
    train_labels = load_mnist_labels(fn)
    # load test images
    fn = os.path.join(path, fn_test_images)
    test_images = load_mnist_images(fn)
    # load test labels
    fn = os.path.join(path, fn_test_labels)
    test_labels = load_mnist_labels(fn)

    return (train_images, train_labels, test_images, test_labels)


'''
def main():
  ds = Dataset()
  #ds.load(CIFAR100, '../cifar-100-python/')
  ds.load(CIFAR10, '../cifar-10-batches-py/')
  ds.set_batch_size(8)
  for i in xrange(3):
    x,y = ds.train_batch()
    #plt.imshow(x[np.random.randint(len(x))])
    #plt.show()
  x,y = ds.test()
  im = pi.fromarray((x[np.random.randint(len(x))]*255).astype(np.uint8), mode="RGB")
  im.show()
  #plt.imshow(x[np.random.randint(len(x))])
  #plt.show()
  #print(str(y[2]))

main()
'''

