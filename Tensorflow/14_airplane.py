"""
비행기가 어떤 조건에서 이륙할 때 얼마만큼의 이륙거리가 필요한지 예측해보자
가상의 비행기 B777-200 이륙데이터를 사용한다.
- 이륙속도 : 290km/h
- 최대비행기 무게 : 300ton
- 필요한 활주거리 : 2000m
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 셋을 섞음
def shuffle_data(x_data, y_data):
    temp_index = np.arange(len(x_data))
    np.random.shuffle(temp_index)
    x_temp = np.zeros(x_data.shape)
    y_temp = np.zeros(y_data.shape)
    x_temp = x_data[temp_index]
    y_temp = y_data[temp_index]
    return x_temp, y_temp

"""
 데이터 정규화
 이륙속도
 무게
 와 같이 서로 다른 단위, 간격을
 일정한 비율로 맞춰주는 작업
 전처리 작업
"""

# 0 ~ 1사이 값으로 정규화
def minmax_normalize(x):
    xmax, xmin = x.max(), x.min()
    norm = (x-xmin)/(xmax-xmin)
    return norm

# 정규화 후 realx에 해당하는 정규값 리턴
def minmax_get_norm(realx, arrx):
    xmax, xmin = arrx.max(), arrx.min()
    normx = (realx - xmin) / (xmax - xmin)
    return normx

# 훈련을 끝내고 원래 단위로 환원할 때 사용
# 0 ~ 1 사이 값을 실제 값으로 역정규값 리턴
def minmax_get_denorm(normx, arrx):
    xmax, xmin = arrx.max(), arrx.min()
    realx = normx * (xmax - xmin) + xmin
    return realx

def main():
    traincsvdata = np.loadtxt('./airplane/trainset.csv',
                              unpack=True, delimiter=',',
                              skiprows=1)   # 제목은 skip
    num_points = len(traincsvdata[0])
    print("points : ", num_points)

    # speed(km/h)
    x1_data = traincsvdata[0]
    # weight(ton)
    x2_data  =traincsvdata[1]
    # distance(m)
    y_data = traincsvdata[2]

    # 빨간색(m) + 둥근점(o)로 시각화
    plt.plot(x1_data, y_data, 'mo')
    plt.suptitle('Training set(x1)', fontsize=16)
    plt.xlabel('speed to take off')
    plt.ylabel('distance')
    plt.show()

    # 파란색(b) + 둥근점(*)로 시각화
    plt.plot(x2_data, y_data, 'b*')
    plt.suptitle('Training set(x2)', fontsize=16)
    plt.xlabel('weight')
    plt.ylabel('distance')
    plt.show()

    x1_data = minmax_normalize(x1_data)
    x2_data = minmax_normalize(x2_data)
    y_data = minmax_normalize(y_data)

    x_data = [[item for item in x1_data],
              [item for item in x2_data]]
    x_data = np.reshape(x_data, 600, order='F')
    x_data = np.reshape(x_data, (-1, 2))
    y_data = np.reshape(y_data, [len(y_data), 1])

    BATCH_SIZE = 5
    BATCH_NUM = int(len(x1_data)/BATCH_SIZE)

    # Placeholder 생성
    input_data = tf.placeholder(tf.float32, shape=[None, 2])
    output_data = tf.placeholder(tf.float32, shape=[None, 1])

    # Weight 생성
    W1 = tf.Variable(tf.random_uniform([2, 5], 0.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([5, 3], 0.0, 1.0))
    W_out = tf.Variable(tf.random_uniform([3, 1], 0.0, 1.0))

    # activation function
    hidden1 = tf.nn.sigmoid(tf.matmul(input_data, W1))
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W2))
    output = tf.matmul(hidden2, W_out)

    # 비용함수 cost, loss
    loss = tf.reduce_mean(tf.square(output - output_data))
    # 최적화 함수
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    # 세션 생성후 training
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    epoch = 1000
    for step in range(epoch):
        index = 0
        x_data, y_data = shuffle_data(x_data, y_data)

        total_cost = 0.
        for batch_iter in range(BATCH_NUM-1):
            feed_dict = {input_data:x_data[index:index+BATCH_SIZE],
                         output_data:y_data[index:index+BATCH_SIZE]}
            l, _ = sess.run([loss, train], feed_dict=feed_dict)
            index += BATCH_SIZE
            total_cost += l
        if step % 50 == 0:
            print("step: ", step, ", Loss: ", total_cost/(BATCH_NUM-1))

    # 학습이 끝난 후 테스트 데이터 입력해봄
    print("# 학습완료. 임의값으로 이륙거리 추정 #")
    arr_ask_x = [[290, 210],
                 [320, 210],
                 [300, 300],
                 [320, 300]]
    for i in range(len(arr_ask_x)):
        ask_x = [arr_ask_x[i]]
        ask_norm_x = [[minmax_get_norm(ask_x[0][0], traincsvdata[0]),
                       minmax_get_norm(ask_x[0][1], traincsvdata[1])]]
        answer_norm_y = sess.run(output,
                                 feed_dict={input_data:ask_norm_x})
        answer_y = minmax_get_denorm(answer_norm_y, traincsvdata[2])
        print("이륙거리계산 - 이륙속도(X1): ", ask_x[0][0], "km/h, ",
              "비행기무게(X2): ", ask_x[0][1], "ton, ",
              "이륙거리(y): ", answer_y[0][0], "m")

    # 테스트 셋 파일 읽음
    test_csv_x_data = np.loadtxt('./airplane/testset_x.csv',
                                 unpack=True, delimiter=',',
                                 skiprows=1)
    test_csv_y_data = np.loadtxt('./airplane/testset_y.csv',
                                 unpack=True, delimiter=',',
                                 skiprows=1)
    # speed(km/h)
    test_x1_data = test_csv_x_data[0]
    # weight(ton)
    test_x2_data = test_csv_x_data[1]

    # 테스트셋 정규화
    test_x1_data = minmax_normalize(test_x1_data)
    test_x2_data = minmax_normalize(test_x2_data)
    test_y_data = minmax_normalize(test_csv_y_data)

    # 검증하기 위한 가공 과정
    test_x_data = [[item for item in test_x1_data],
                   [item for item in test_x2_data]]
    test_x_data = np.reshape(test_x_data, len(test_x1_data)*2, order='F')
    test_x_data = np.reshape(test_x_data, (-1, 2))

    plt.plot(list(range(len(test_csv_y_data))), test_csv_y_data, 'mo')
    feed_dict = {input_data:test_x_data}
    test_pred_y_data = minmax_get_denorm(sess.run(output,
                                                  feed_dict=feed_dict),
                                         traincsvdata[2])
    plt.plot(list(range(len(test_csv_y_data))), test_pred_y_data, 'k*')

    #그래프 표시
    plt.suptitle('Test Result', fontsize=16)
    plt.xlabel('index(x1, x2)')
    plt.ylabel('distance')
    plt.show()
    pass


if __name__ == '__main__':
    main()
















