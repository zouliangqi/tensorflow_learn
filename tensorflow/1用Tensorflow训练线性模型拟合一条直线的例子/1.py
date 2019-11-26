# 加载tensorflow和numpy两个模块
import tensorflow as tf
import numpy as np
 
# creat data
'''使用 numpy 来创建我们的数据。
接着, 我们用 tf.Variable 来创建描述 y 的参数. 
我们可以把 y_data = x_data*0.1 + 0.3 想象成 y=Weights * x + biases, 
然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3'''
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
 
# create tensorflow structure start
'''搭建模型'''
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
 
y = Weights*x_data +biases
 
'''计算误差，均方误差函数'''
loss = tf.reduce_mean(tf.square(y - y_data))
'''0.5 is learning rate。反向传递误差的工作就教给optimizer了,
 我们使用的误差传递方法是梯度下降法: Gradient Descent。
 然后我们使用 optimizer 来进行参数的更新.'''
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
 
'''到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构. 
在使用这个结构之前, 我们必须先初始化所有之前定义的Variable, 所以这一步是很重要的!'''
#init = tf.initialize_all_variables() # tf马上就要废弃这种写法
init = tf.global_variables_initializer() # 替换成这样就好
# create tensorflow structure end
 
'''接着,我们再创建会话 Session. 我们用 Session 来执行 init 初始化步骤.'''
sess = tf.Session()
sess.run(init)     # Very important
'''用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.'''
for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(Weights),sess.run(biases))
