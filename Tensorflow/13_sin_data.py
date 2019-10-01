import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 셋을 섞음
def shuffle_data(x_train, y_train):
    temp_index = np.arange(len(x_train))

    np.random.shuffle(temp_index)

    x_temp = np.zeros(x_train.shape)
    y_temp = np.zeros(y_train.shape)
    x_temp = x_train[temp_index]
    y_temp = y_train[temp_index]

    return x_temp, y_temp

def make_set():
    num_points = 5000
    vectors_set = []
    for i in range(num_points):
        x1 = np.random.normal(.0, 1.0)
        y1 = np.sin(x1) + np.random.normal(0.0, 0.1)
        vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

    plt.plot(x_data, y_data, 'go')
    plt.show()

    return x_data, y_data

def main():
    x_data, y_data = make_set()

    # 배치 수행단위
    BATCH_SIZE = 100
    BATCH_NUM = int(len(x_data)/BATCH_SIZE)

    # 데이터를 세로로(한 개씩 ) 나열한 형태로 reshape
    x_data = np.reshape(x_data, [len(x_data), 1])
    y_data = np.reshape(y_data, [len(y_data), 1])

    # 총 개수는 ??, 1개씩 전달되는 Placeholder 생성
    input_data = tf.placeholder(tf.float32, shape=[None, 1])
    output_data = tf.placeholder(tf.float32, shape=[None, 1])

    # 레이어 간 Weight 정의 후 랜덤 초기화
    # W1 = tf.Variable(tf.random_uniform([1, 5], -1.0, 1.0))
    # W2 = tf.Variable(tf.random_uniform([5, 3], -1.0, 1.0))
    # W3 = tf.Variable(tf.random_uniform([3, 1], -1.0, 1.0))
    #
    # # 레이어와 입력값의 내적계산(행렬곱, 점곱)
    # # 비선형성을 주는 sigmoid 추가
    # hidden1 = tf.nn.sigmoid(tf.matmul(input_data, W1))
    # hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W2))
    # # 일반적으로 마지막 출력층은 activation Function을 적용하지 않는다.
    # output = tf.matmul(hidden2, W3)

    # 선형
    # W1 = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0))
    # output = tf.matmul(input_data, W1)

    # hidden layer 2계층
    W1 = tf.Variable(tf.random_uniform([1, 5], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0))
    hidden1 = tf.nn.sigmoid(tf.matmul(input_data, W1))
    output = tf.matmul(hidden1, W2)

    # cost(loss)-비용함수, 최적화함수, train
    loss = tf.reduce_mean(tf.square(output - output_data))
    # optimizer = tf.train.GradientDescentOptimizer(0.01)
    # optimizer = tf.train.RMSPropOptimizer(0.01)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 5000회 학습을 반복하며 값 업데이트
    for step in range(5001):
        index = 0
        x_data, y_data = shuffle_data(x_data, y_data)

        # 배치 크기만큼 학습을 진행
        for batch_iter in range(BATCH_NUM - 1):
            feed_dict = {input_data:x_data[index:index+BATCH_SIZE],
                         output_data:y_data[index:index+BATCH_SIZE]}
            sess.run(train, feed_dict = feed_dict)
            index += BATCH_SIZE

        if (step % 100 == 0 or (step < 100 and step % 10 == 0)):
            print("Step=%5d, Loss=%f" % (step,
                                         sess.run(loss,
                                                  feed_dict=feed_dict)))

    feed_dict = {input_data:x_data}

    plt.plot(x_data, y_data, 'go')
    plt.plot(x_data, sess.run(output, feed_dict=feed_dict), 'k*')
    plt.xlabel('x')
    plt.xlim(-4, 3)
    plt.ylabel('y')
    plt.ylim(-1.5, 1.5)
    plt.show()

if __name__ == '__main__':
    main()