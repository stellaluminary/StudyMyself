import tensorflow as tf

b1 = tf.Variable(0.0, name="bias")

saver = tf.train.Saver()

sess = tf.Session()

# 읽어들일때는 아래처럼 초기화하지 않는다.
# sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state("./saver_bias/")
if tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Variable is restored")

print("bias: ", sess.run(b1))

