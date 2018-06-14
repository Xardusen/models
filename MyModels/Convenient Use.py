import tensorflow as tf

a = tf.constant(25,name='input1')
b = tf.constant(99, name='number99')
# b = tf.Variable(tf.random_uniform([3]),name='input2')
add = tf.add_n([a,b])
# with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
# print(tf.Session().run(add))print(tf.Session().run(add))
print(tf.Session().run(a))
