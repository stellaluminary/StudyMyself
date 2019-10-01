import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# 학습 시에는 dropout을 사용, 테스트 시에는 dropout을 사용하지 않는다.
is_training = tf.placeholder(tf.bool)

# 입력값은 X, 32개의 Filter, Filter 크기는 3 x 3
L1 = tf.layers.conv2d(X, 32, [3, 3])
# 입력값은 L1, ksize 2 x 2, strides 2 x 2
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
# 입력값은 L1, dropout 70%, is_trainng:True면 적용, False면 미적용
L1 = tf.layers.dropout(L1, 0.7, is_training)

L2 = tf.layers.conv2d(L1, 64, [3, 3])
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)

# 이전에 다차원 행렬을 1차원으로 변환
L3 = tf.contrib.layers.flatten(L2)
# 입력값은 L3로 하고, Node수를 256, 활성화 함수:relu
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, is_training)

model = tf.layers.dense(L3, 10, activation=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epoch_num = 20
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(epoch_num):      # 전체를 몇번 학습
    total_cost = 0

    for i in range(total_batch):    # 전체데이터를 batch_size로 쪼개서 전체학습
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X:batch_xs, Y:batch_ys,
                                          is_training:True})
        total_cost += cost_val
    
    print('Epoch: ', '%4d' % (epoch+1),
          'Avg. Cost = ', '{:.4f}'.format(total_cost / total_batch))
print("최적화 완료")

# 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: ", sess.run(accuracy,
                    feed_dict={X:mnist.test.images.reshape(-1, 28, 28, 1),
                                Y:mnist.test.labels,
                                is_training:False}))


sess.close()













