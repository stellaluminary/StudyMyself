"""
Fine Tuning:
고양이를 잘 학습된 데이터 + 호랑이 이미지
=> 조금만 훈련시켜도 잘 될 가능성이 높다
이미 잘 특징이 뽑히도록 훈련된 데이터를 이용해서
비교적 유사한 새로운 데이터를 학습하는 방법

전이 학습:
임의의 값으로 초기화한 파라미터로 학습하는 것보다
훨씬 빠른 시간에 학습이 이루어진다.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MNIST 데이터를 다운로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./tmp/data/", one_hot=True)

# 학습에 필요한 설정값 정의
learning_rate_RMSProp = 0.02        # AutoEncoder용
learning_rate_GradientDescent = 0.5 # Softmax 분류용
num_epochs = 100                    # 반복 회수
batch_size = 256                    # 배치 사이즈(학습 단위)
display_step = 1                    # 몇 step마다 화면 log 출력
input_size = 784                    # 28 x 28
hidden1_size = 128
hidden2_size = 64

# AutoEncoder 함수
def build_autoencoder(x):
    # Encoding 784 -> 128 -> 64 : Noise를 제거하고 압축특징을 추출
    Wh_1 = tf.Variable(tf.random_normal([input_size, hidden1_size]))
    bh_1 = tf.Variable(tf.random_normal([hidden1_size]))
    H1_output = tf.nn.sigmoid(tf.matmul(x, Wh_1) + bh_1)
    Wh_2 = tf.Variable(tf.random_normal([hidden1_size, hidden2_size]))
    bh_2 = tf.Variable(tf.random_normal([hidden2_size]))
    H2_output = tf.nn.sigmoid(tf.matmul(H1_output, Wh_2) + bh_2)
    # Decoding 64 -> 128 -> 784
    Wh_3 = tf.Variable(tf.random_normal([hidden2_size, hidden1_size]))
    bh_3 = tf.Variable(tf.random_normal([hidden1_size]))
    H3_output = tf.nn.sigmoid(tf.matmul(H2_output, Wh_3))
    Wo = tf.Variable(tf.random_normal([hidden1_size, input_size]))
    bo = tf.Variable(tf.random_normal([input_size]))
    X_reconstructed = tf.nn.sigmoid(tf.matmul(H3_output, Wo) + bo)
    return X_reconstructed, H2_output

# Softmax 분류기 함수
# 64개로 압축된 특성을 10개로 분류한다.
def build_softmax_classifier(x):
    # 64 -> 10
    W_softmax = tf.Variable(tf.random_normal([hidden2_size, 10]))
    b_softmax = tf.Variable(tf.random_normal([10]))
    y_pred = tf.nn.softmax(tf.matmul(x, W_softmax) + b_softmax)
    return y_pred

# main 함수
def main():
    x = tf.placeholder(tf.float32, shape=[None, input_size])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    # AutoEncoder를 통해 기대값과 압축특징을 받는다
    y_pred, extacted_features = build_autoencoder(x)
    y_true = x

    # Softmax 분류기, 64개의 압축특징을 넣어준다.
    y_pred_softmax = build_softmax_classifier(extacted_features)
    
    # 1. Pre-Training: MNIST의 압축 후 복원이 잘되도록
    pretraining_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    pretraining_train_step = \
        tf.train.RMSPropOptimizer(learning_rate_RMSProp).minimize(pretraining_loss)
    # 2. Fine-Tuning : Softmax분류가 잘되도록 학습
    # Cross-Entropy loss 함수
    finetuning_loss = \
        tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred_softmax),
                                      reduction_indices=[1]))
    finetuning_train_step = \
        tf.train.GradientDescentOptimizer(
            learning_rate_GradientDescent).minimize(finetuning_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch = int(mnist.train.num_examples / batch_size)
        # step1 : AutoEncoder 최적화 학습, 압축특성을 잘 뽑아내는 학습
        for epoch in range(num_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, preLoss = \
                    sess.run([pretraining_train_step, pretraining_loss],
                                      feed_dict={x:batch_xs})
            if epoch % display_step == 0:
                print("Epoch: %d, PreLoss: %f" % (epoch+1, preLoss))
        print("AutoEncoder Training Complete!!!")
        
        # step2 : 압축특징 + Softmax 최적화 학습
        for epoch in range(num_epochs + 100):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, fineLoss = \
                    sess.run([finetuning_train_step, finetuning_loss],
                             feed_dict={x:batch_xs, y:batch_ys})
            if epoch % display_step == 0:
                print("Epoch: %d, findLoss: %f" % (epoch+1, fineLoss))
        print("Softmax Classifier Training Complete!!!")
        
        # step3 : 정확도 출력
        is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred_softmax, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print("AutoEncoder + Softmax => accuracy: %f" % sess.run(
            accuracy, feed_dict={x:mnist.test.images,
                                 y:mnist.test.labels}))
        
    pass

if __name__ == '__main__':
    main()
















