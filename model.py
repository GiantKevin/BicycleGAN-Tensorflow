import os
import random

from tqdm import trange, tqdm
from scipy.misc import imsave
import tensorflow as tf
import numpy as np

from generator import Generator
from encoder import Encoder
from discriminator import Discriminator
from utils import logger


class BicycleGAN(object):
    def __init__(self, args):
        self._log_step = args.log_step # 日志保存
        self._batch_size = args.batch_size # batch_size 大小
        self._image_size = args.image_size # 图片大小
        self._latent_dim = args.latent_dim # 噪声的维度
        self._coeff_gan = args.coeff_gan # gan的常量系数
        self._coeff_vae = args.coeff_vae # vae的常量系数
        self._coeff_reconstruct = args.coeff_reconstruct # 重建函数的常量系数
        self._coeff_latent = args.coeff_latent # 潜在编码空间常量系数
        self._coeff_kl = args.coeff_kl # kl损失函数常量系数
        self._norm = 'instance' if args.instance_normalization else 'batch' # 选择norm的格式
        self._use_resnet = args.use_resnet # 是否使用resnet结构

        self._augment_size = self._image_size + (30 if self._image_size == 256 else 15) # 改变图像大小？
        self._image_shape = [self._image_size, self._image_size, 3] # 设置输入图片的形状
 
        self.is_train = tf.placeholder(tf.bool, name='is_train') # 设定是否训练
        self.lr = tf.placeholder(tf.float32, name='lr') # 设定学习速率
        self.global_step = tf.train.get_or_create_global_step(graph=None) 

        image_a = self.image_a = \ # 设定图片a,作为草图
            tf.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_a')
        image_b = self.image_b = \ # 设定图片b,作为结果图
            tf.placeholder(tf.float32, [self._batch_size] + self._image_shape, name='image_b')
        z = self.z = \ # 编码器生成的潜在空间向量z,可以是真实图片进过编码器的结果，也可以是随机采样的标准正态分布
            tf.placeholder(tf.float32, [self._batch_size, self._latent_dim], name='z')

        # Data augmentation # 对图片进行数据增强
        seed = random.randint(0, 2**31 - 1)  # 生成随机种子
        def augment_image(image):
            image = tf.image.resize_images(image, [self._augment_size, self._augment_size]) # 对输入图片进行重新设定大小
            image = tf.random_crop(image, [self._batch_size] + self._image_shape, seed=seed) # 对图片进行随机裁剪
            image = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, seed), image) # 对图片进行左右反转
            return image

        image_a = tf.cond(self.is_train, # 图像a合集
                          lambda: augment_image(image_a),
                          lambda: image_a)
        image_b = tf.cond(self.is_train, # 图像b合集
                          lambda: augment_image(image_b),
                          lambda: image_b)

        # Generator 生成器
        G = Generator('G', is_train=self.is_train,
                      norm=self._norm, image_size=self._image_size)

        # Discriminator 鉴别器
        D = Discriminator('D', is_train=self.is_train,
                          norm=self._norm, activation='leaky',
                          image_size=self._image_size)

        # Encoder 编码器
        E = Encoder('E', is_train=self.is_train,
                    norm=self._norm, activation='relu',
                    image_size=self._image_size, latent_dim=self._latent_dim,
                    use_resnet=self._use_resnet)

        # conditional VAE-GAN: B -> z -> B' 
        z_encoded, z_encoded_mu, z_encoded_log_sigma = E(image_b) # 对真实图片进行编码，产生z
        image_ab_encoded = G(image_a, z_encoded) # 对输入图片和编码器输入生成器，产生假图片img_ab_encoed

        # conditional Latent Regressor-GAN: z -> B' -> z'
        image_ab = self.image_ab = G(image_a, z) # 将输入图片和随机采样的标准正态分布输入生成器，产生假图片img_ab
        z_recon, z_recon_mu, z_recon_log_sigma = E(image_ab) # 将计算图片img_ab输入Encoder产生z_recon


        # Discriminate real/fake images # 鉴别器对图片进行鉴别
        D_real = D(image_b) # 鉴别真的图片
        D_fake = D(image_ab) # 鉴别随机采样的z和img产生的假图片
        D_fake_encoded = D(image_ab_encoded) # 鉴别将真图片输入编码器的z生成的假图片

        loss_vae_gan = (tf.reduce_mean(tf.squared_difference(D_real, 0.9)) +
            tf.reduce_mean(tf.square(D_fake_encoded))) # 计算vae_gan的损失

        loss_image_cycle = tf.reduce_mean(tf.abs(image_b - image_ab_encoded)) # 计算真实图片和编码产生的假图片之间的l1损失

        loss_gan = (tf.reduce_mean(tf.squared_difference(D_real, 0.9)) +
            tf.reduce_mean(tf.square(D_fake))) # 计算gan的损失

        loss_latent_cycle = tf.reduce_mean(tf.abs(z - z_recon)) # 计算z和重构的z_recon之间的l1损失

        loss_kl = -0.5 * tf.reduce_mean(1 + 2 * z_encoded_log_sigma - z_encoded_mu ** 2 -
                                       tf.exp(2 * z_encoded_log_sigma)) # 计算encoded产生的z和标准的正态分布之前的kl散度

        loss = self._coeff_vae * loss_vae_gan - self._coeff_reconstruct * loss_image_cycle + \
            self._coeff_gan * loss_gan - self._coeff_latent * loss_latent_cycle - \
            self._coeff_kl * loss_kl

        # Optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops): # 设置优化器
            self.optimizer_D = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                                .minimize(loss, var_list=D.var_list, global_step=self.global_step)
            self.optimizer_G = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                                .minimize(-loss, var_list=G.var_list)
            self.optimizer_E = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                                .minimize(-loss, var_list=E.var_list)

        # Summaries
        self.loss_vae_gan = loss_vae_gan
        self.loss_image_cycle = loss_image_cycle
        self.loss_latent_cycle = loss_latent_cycle
        self.loss_gan = loss_gan
        self.loss_kl = loss_kl
        self.loss = loss

        tf.summary.scalar('loss/vae_gan', loss_vae_gan)
        tf.summary.scalar('loss/image_cycle', loss_image_cycle)
        tf.summary.scalar('loss/latent_cycle', loss_latent_cycle)
        tf.summary.scalar('loss/gan', loss_gan)
        tf.summary.scalar('loss/kl', loss_kl)
        tf.summary.scalar('loss/total', loss)
        tf.summary.scalar('model/D_real', tf.reduce_mean(D_real))
        tf.summary.scalar('model/D_fake', tf.reduce_mean(D_fake))
        tf.summary.scalar('model/D_fake_encoded', tf.reduce_mean(D_fake_encoded))
        tf.summary.scalar('model/lr', self.lr)
        tf.summary.image('image/A', image_a[0:1])
        tf.summary.image('image/B', image_b[0:1])
        tf.summary.image('image/A-B', image_ab[0:1])
        tf.summary.image('image/A-B_encoded', image_ab_encoded[0:1])
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, data_A, data_B):
        logger.info('Start training.')
        logger.info('  {} images from A'.format(len(data_A)))
        logger.info('  {} images from B'.format(len(data_B)))

        assert len(data_A) == len(data_B), \
            'Data size mismatch dataA(%d) dataB(%d)' % (len(data_A), len(data_B))
        data_size = len(data_A) # 图片的总数目
        num_batch = data_size // self._batch_size # 总batch的数目
        epoch_length = num_batch * self._batch_size # epoch的长度

        num_initial_iter = 8 # z_dim
        num_decay_iter = 2 
        lr = lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter # lr的衰减速率

        initial_step = sess.run(self.global_step) 
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length # 总共迭代的数目
        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        for step in t:
            #TODO: resume training with global_step
            epoch = step // epoch_length  # 迭代的epoch次数
            iter = step % epoch_length # 计算第几次迭代

            if epoch > num_initial_iter: 
                lr = max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay) # 计算lr的输入

            if iter == 0: # 对数据进行shuffle
                data = zip(data_A, data_B) 
                random.shuffle(data)
                data_A, data_B = zip(*data)

            image_a = np.stack(data_A[iter*self._batch_size:(iter+1)*self._batch_size]) # 输入图片image_a
            image_b = np.stack(data_B[iter*self._batch_size:(iter+1)*self._batch_size]) # 输入图片image_b
            sample_z = np.random.normal(size=(self._batch_size, self._latent_dim)) # 采样的z

            fetches = [self.loss, self.optimizer_D,
                       self.optimizer_G, self.optimizer_E] # 损失合集
            if step % self._log_step == 0:
                fetches += [self.summary_op]

            fetched = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                   self.image_b: image_b,
                                                   self.is_train: True,
                                                   self.lr: lr,
                                                   self.z: sample_z})

            if step % self._log_step == 0:
                z = np.random.normal(size=(1, self._latent_dim)) # 输入噪声z
                image_ab = sess.run(self.image_ab, feed_dict={self.image_a: image_a,
                                                            self.z: z,
                                                            self.is_train: False}) # 噪声产生的图片image_ab
                imsave('results/r_{}.jpg'.format(step), np.squeeze(image_ab, axis=0)) # 产生一个维度

                summary_writer.add_summary(fetched[-1], step)
                summary_writer.flush()
                t.set_description('Loss({:.3f})'.format(fetched[0]))

    def test(self, sess, data_A, data_B, base_dir):
        step = 0
        for (dataA, dataB) in tqdm(zip(data_A, data_B)):
            step += 1
            image_a = np.expand_dims(dataA, axis=0) # image_a
            image_b = np.expand_dims(dataB, axis=0) # image_b
            images_random = []
            images_random.append(image_a) 
            images_random.append(image_b)
            images_linear = []
            images_linear.append(image_a)
            images_linear.append(image_b)

            for i in range(23):
                z = np.random.normal(size=(1, self._latent_dim))
                image_ab = sess.run(self.image_ab, feed_dict={self.image_a: image_a,
                                                        self.z: z,
                                                        self.is_train: False})
                images_random.append(image_ab)

                z = np.zeros((1, self._latent_dim))
                z[0][0] = (i / 23.0 - 0.5) * 2.0
                image_ab = sess.run(self.image_ab, feed_dict={self.image_a: image_a,
                                                        self.z: z,
                                                        self.is_train: False})
                images_linear.append(image_ab)

            image_rows = []
            for i in range(5):
                image_rows.append(np.concatenate(images_random[i*5:(i+1)*5], axis=2))
            images = np.concatenate(image_rows, axis=1)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'random_{}.jpg'.format(step)), images)

            image_rows = []
            for i in range(5):
                image_rows.append(np.concatenate(images_linear[i*5:(i+1)*5], axis=2))
            images = np.concatenate(image_rows, axis=1)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'linear_{}.jpg'.format(step)), images)
