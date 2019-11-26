import tensorflow as tf
res = tf.random_uniform((4, 4), -1, 1)
with tf.Session() as sess:
    print(sess.run(res))
"""tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))
返回4*4的矩阵，产生于low和high之间，产生的值是均匀分布的。
"""
