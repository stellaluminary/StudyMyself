import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 신경망 모델 구성
# batch_size, height, width, channel => 28 x 28 x Gray
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
# 학습시에는 1.0보다 작게, 테스트시에는 1.0으로(Fully Connect Network)
# dropout 사용하려고
keep_prob = tf.placeholder(tf.float32)

# Filter에 따른 가중치 Filter크기 3 x 3 x channel=1, Filter개수 32개
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# strides는 옆으로 이동
# batch, height, width, channel
# index0과 index3은 사용하지 않는다.
# 옆으로 1칸, 아래로 1칸
# 만약 2칸 stride하고 싶으면 [1, 2, 2, 1]
# strides가 1일 때 zero-padding을 적용해서 원래 크기와 동일하게 만들어준다
# strides가 2이면 절반 크기로 만들어준다.
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') # 28 x 28
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')  # 14 x 14

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') # 14 x 14
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME') # 7 x 7

W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)
epoch_num = 15

for epoch in range(epoch_num):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X:batch_xs,
                                          Y:batch_ys,
                                          keep_prob:0.7})
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch + 1),
          'Avg. Cost: ', '{:.3f}'.format(total_cost / total_batch))
print("*" * 10, " Training Complete!!! ", "*" * 10)

# 정확도 테스트
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: ', sess.run(accuracy,
            feed_dict={X:mnist.test.images.reshape(-1, 28, 28, 1),
                       Y:mnist.test.labels,
                       keep_prob:1}))

sess.close()










