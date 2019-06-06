import numpy as np

char_inputs = np.zeros([2, 1, 3],
                       np.int32)
res=np.vstack([[1,2,3]]+[[4,5,6]])

# char_inputs[0,0,:]=res[0:1]
# print(res)#[[1, 2, 3], [4, 5, 6]]
# print(res+1)
import  tensorflow as tf

with tf.Session() as sess:
    z=tf.random_normal((3,4,2),mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
    f=tf.unstack(z, axis=1)

    print(sess.run([z,f]))
