import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

total_epoch = 100       # 총 반복 회수
batch_size = 100        # 1회 학습 분량
learning_rate = 0.0002
n_hidden = 256
n_input = 784           # 28 x 28
n_noise = 128           # 임의의 데이터 -> 학습 -> MNIST

# 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, n_input]) # MNIST Real Data - Real
Z = tf.placeholder(tf.float32, [None, n_noise]) # Random Noise    - Fake

# Generative  - Fake MNIST
# 위조 지폐 제작
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# Discriminator - Real or Fake
# 경찰이 진짜 화폐인지, 위조 지폐인지 구분
# 1이면 Real, 0이면 Fake
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# Fake MNIST 생성
# 위조 지폐 제작
# 128 -> 256 -> 784
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output

# noise값을 생성하는 함수
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# 진짜면 1, 가짜면 0으로 반환하는 구분 함수
# 784 -> 256 -> 1
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output


G = generator(Z)            # Make fake MNIST
D_gene = discriminator(G)   # fake MNIST -> 가짜로 판정내리도록 학습 => 0
D_real = discriminator(X)   # real MNIST -> 진짜로 판정내리도록 학습 => 1

# loss(cost) 함수
# tf.log(1) = 0
# tf.log(0) = -무한
# loss_D = 0으로 하려면 D_real = 1이 되도록, D_gene = 0이 되도록 학습
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
# loss_G = 1되도록 해야 진짜 같은 위조 지폐가 된다.
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_W2, D_b2] # 구분자와 연결된 학습 변수들 리스트
G_var_list = [G_W1, G_b1, G_W2, G_b2] # 생성자와 연결된 학습 변수들 리스트

# train 함수(optimizer함수 포함)
# 음수 값 -> 0으로 나오도록 학습을 시켜야 하므로 loss값에 -를 붙여서
# maximize를 유도한다. maximize가 없어서
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                                                         var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,
                                                         var_list=G_var_list)

# GAN 신경망 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # real MNIST
        noise = get_noise(batch_size, n_noise)                  # noise

        # loss_val_D가 점점 0에 가까워지는 방향으로 학습
        # 경찰이 진짜 화폐와 위조 화폐를 잘 구분하는 방향으로 학습
        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X:batch_xs, Z:noise})
        # loss_val_G가 점점 0에 가까워지는 방향으로 학습
        # 위폐범이 위조 화폐를 진짜처럼 만드는 방향으로 학습
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Z:noise})

    print('Epoch : ', '%04d' % epoch,
          'D loss : {:.4}'.format(loss_val_D),
          'G loss : {:.4}'.format(loss_val_G))


    # 확인용 가짜 이미지 생성
    # 10번 epoch마다 점점 이미지가 개선되는지 보고 싶으니까 이미지 생성
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)     # noise
        samples = sess.run(G, feed_dict={Z:noise})  # noise -> Fake MNIST

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)),
                                            bbox_inches = 'tight')
        plt.close(fig)

print('*' * 15, " 최적화 완료 ", '*' * 15)
sess.close()













