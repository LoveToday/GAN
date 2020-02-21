'''
简介
生成式对抗网络(GAN, Generative Adversarial Networks) 是一种深度学习模型，是近年来复杂分布上无监督学习最具前景的方法之一
GAN是“生成对抗网络” （Generative Adversarial Networks）的简称，由2014年还在蒙特利尔读博士的lanGoodfellow引入深度学习领域。
2016年，GAN热潮席卷AI领域顶级会议，从ICLR到NIPS，大量高质量论文被发表和探讨。Yann LeCun曾评价GAN是“20年来机器学习领域最酷的想法”
机器学习的模型可大体分为两类，生成模型（Generative Model）和 判别模型（Discriminative Model）. 判别模型需要输入变量，通过某种模型预测。
生成模型是给定某种隐含信息，来随机产生观测数据。
GAN 主要包括了两部分，即生成器generator 与 判别器 discriminator 。 生成器主要用来学习真实图像分布从而让自身生成的图像更加真实，以骗过判别器。
判别器则需要对接收的图片进行真假判别
'''
'''
原理
使得判别器无法判断，无论对于真假样本，输出结果概率都是0.5
在训练过程中，生成器努力地让生成的图像更加真实，而判别器则努力地去识别出图像的真假，
这个过程相当于一个二人博弈，随着时间的推移，生成器和判别器在不断的进行对抗，最终两个网络达到了一个动态均衡；
生成器生成的图像接近于真实图像分布，而判别器识别不出真假图像，对于给定图像的预测为真的概率基本接近0.5
举例
造假币的团伙相当于生成器，他们想通过伪造金钱来骗过银行，使得假币能够正常交易，而银行相当于判别器，需要判断进来的钱是真钱还是假币。
因此假币团伙的目的是要造出银行识别不出的假币而骗过银行，银行则要想办法准确地识别出假币
GAN原理的总结
对于给定的真是图片，判别器是要为其打上标签1
对于给定的生成图片，判别器是要为打上标签0
对于生成器传给判别器的生成图片，生成器希望辨别器打上标签1
训练过程中，生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D。而D的目标就是尽量把G生成的图片和真实的图片分别开来
这样，G和D构成了一个动态的“博弈过程”
博弈的结果是什么？
在最理想的状态下，G可以生成足以“以假乱真”的图片G(z)。 对于D来说，它难以判断G生成的图片究竟是不是真实的，因此D(G(z))= 0.5
'''

# 目的： 得到一个生成模型G， 可以用来生成图片。

# GAN的应用领域
# 一、图像生成
# 二、图像增强
# 三、风格化
# 四、艺术的图像创造


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
import numpy as np 
import os 

class GanHandleImage(object):
    def __init__(self):
        # 初始化一个长度为100的变量用于初始化图片向量
        self.randomLength = 100
        # (self.train_images, train_labels)的shape是(60000, 28, 28) (60000,)
        (self.train_images, self.train_labels), (_, _) = keras.datasets.mnist.load_data()

        # 类型的 train_images 类型 uint8-> float32  用self.train_images.dtype进行查看
        self.train_images = tf.cast(self.train_images, tf.float32)

        self.train_images = np.expand_dims(self.train_images, -1)

        # 对train_images的取值返回映射到[-1. 1]
        self.train_images = self.train_images/127.5 - 1

        self.BATCH_SIZE = 256
        self.BUFFER_SIZE = len(self.train_images)
        
        # 将真实的训练数据转换成TensorSliceDataset
        self.datasets = tf.data.Dataset.from_tensor_slices(self.train_images)
        # 将真实的数据做一个乱序 并且处理成 BatchDataset
        self.datasets = self.datasets.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        
        # loss
        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        # 生成器的 optimazer
        self.generator_optimizer = keras.optimizers.Adam(1e-4)
        # 辨别器的optimazer
        self.discriminator_optimizer = keras.optimizers.Adam(1e-4)

        self.EPOCHS = 30
        # 噪音的向量长度
        self.noise_dim = 100
        # 每个epochs生成的样本个数
        self.num_exp_to_generate = 3
        # 生成num_exp_to_generate个随机向量
        self.seed = tf.random.normal([self.num_exp_to_generate, self.noise_dim])

        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()

    # 生成器
    def generator_model(self):
        model = tf.keras.Sequential()

        model.add(layers.Dense(256, input_shape=(self.randomLength,), use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(512, use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(1024, use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(self.train_images.shape[1]*self.train_images.shape[2]*self.train_images.shape[3], 
                                use_bias=False, activation='tanh'))
        model.add(layers.BatchNormalization())

        model.add(layers.Reshape(self.train_images.shape[1:]))

        return model
    # 判别器
    def discriminator_model(self):
        model = keras.Sequential()

        model.add(layers.Flatten())

        model.add(layers.Dense(512, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(256, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Dense(1, use_bias=False, activation='sigmoid'))

        return model 
    # 判别器
    def discriminator_loss(self, real_out, fake_out):
        real_loss = self.cross_entropy(tf.ones_like(real_out), real_out)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
        return real_loss + fake_loss

    # 生成器
    def generator_loss(self, fake_out):
        return self.cross_entropy(tf.ones_like(fake_out), fake_out)

    def train_step(self, images):
        # 长度为self.noise_dim的随机数
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            real_out = self.discriminator(images, training=True)

            gen_image = self.generator(noise, training=True)
            fake_out = self.discriminator(gen_image, training=True)

            gen_loss = self.generator_loss(fake_out)
            disc_loss = self.discriminator_loss(real_out, fake_out)

        gradient_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradient_disc = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradient_gen, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradient_disc, self.discriminator.trainable_variables))


    def generate_plot_image(self, gen_model, test_noise, epoch):
        pre_images = gen_model(test_noise, training=False)
        fig = plt.figure(figsize=(3,1))
        for i in range(pre_images.shape[0]):
            plt.subplot(1,3,i+1)
            plt.imshow((pre_images[i, :, :, 0] + 1)/2, cmap='gray')
            plt.axis('off')
        fig.savefig('./images/{}.png'.format(epoch))
        # plt.show()
    

    def train(self):
        for epoch in range(self.EPOCHS):
            for image_batch in self.datasets:
                self.train_step(image_batch)
                print('.', end='')
            self.generate_plot_image(self.generator, self.seed, epoch)



if __name__ == '__main__':
    if not os.path.exists('./images'):
        os.makedirs('./images')
    GanHandleImage().train()



