import tensorflow as tf
import numpy as np
import optimizer as om
import dataset as ds
import matplotlib.pyplot as plt
from PIL import Image as pi
import model
import time
import os

EPOCH_NUM = 500

def main():
    print('Loading MNIST dataset...')
    mnist = ds.Dataset()
    mnist.load(ds.MNIST, '../MNIST/')
    model_save_dir = './model_trained/'
    train_log_dir = './train_log/'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.exists(train_log_dir):
        os.mkdir(train_log_dir)
    batch_size = 3
    mnist.set_batch_size(batch_size)
    t_x, t_labels, t_gt_labels, t_err = model.mnist_model(mnist.shape_of_sample(), [16,32,16], mnist.shape_of_label())
    batch_num = int(EPOCH_NUM * len(mnist.train_samples)/batch_size)
    # create an optimizer for training
    opt = om.Optimizer()
    minimizer = opt.minimize(t_err)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('loss', t_err)
    merge_summary = tf.summary.merge_all()  
    train_writer = tf.summary.FileWriter(train_log_dir,s ess.graph) 
      
    for i in xrange(batch_num):
        x, gt_labels = mnist.train_batch()
        err,_,summary = sess.run([t_err, minimizer, merge_summary], 
            feed_dict={t_x:x, t_gt_labels:gt_labels.astype(np.int32)})
        minimizer = opt.update(sess, err)
        if i%100==0:
            print('batch=%d epoch=%d err=%.6f'%(i,int(i*batch_size/len(mnist.train_samples)), err))
            train_writer.add_summary(summary, int(i/100))
        if minimizer == None:
            model_path = os.path.join(
                    model_save_dir, 
                    time.strftime('Y%Ym%md%dH%HM%MS%S', 
                        time.localtime(int(time.time()))))
            print('final model saved to %s'%model_path)
            saver.save(sess, model_path)
            exit(0)
        else:
            if i%10000==0:
                model_path = os.path.join(
                        model_save_dir, 
                        time.strftime('Y%Ym%md%dH%HM%MS%S', 
                            time.localtime(int(time.time()))))
                print('saving model to %s'%model_path)
                saver.save(sess, model_path)

    '''    
    x,y = mnist.test()
    rand_id = np.random.randint(len(x))
    plt.imshow(x[rand_id].reshape([x.shape[1], x.shape[2]]))
    plt.show()
    print(str(y[rand_id]))
    '''
main()
