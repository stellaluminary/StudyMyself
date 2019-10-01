import tensorflow as tf

b1 = tf.Variable(2.0, name="bias")

# Saver 생성 (학습데이터를 저장하기 위한 객체)
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("save test bias : ", sess.run(b1))

# 변수 저장
save_path = saver.save(sess, "./saver_bias/bias.ckpt")
print("Model saved in file: %s" % save_path)
sess.close()