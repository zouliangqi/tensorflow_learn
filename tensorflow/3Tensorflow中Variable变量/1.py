import tensorflow as tf
 
state = tf.Variable(3, name="counter")
# print(state.name)
# 定义常量one
one = tf.constant(1)
# 定义加法步骤（注：此步骤并没有直接计算）
new_value = tf.add(state , one)
# 将State更新为new_value
update = tf.assign(state,new_value)
# 如果在Tensorflow中定义了变量，那么初始化变量是最重要的，所以定义变量以后一定要定义init = tf.initialize_all_variables() 
init = tf.initialize_all_variables() # must have if define variable
 
with tf.Session() as sess:
    # 变量还没有激活，需要在sess里激活init
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
