import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
from scipy.misc import logsumexp
from sklearn import mixture 
from numpy import *
import scipy
import os
import shutil
output_path = "output_distribution"
to_restore =False
sns.set(color_codes=True)
s = np.random.uniform(1,0,10000)
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

Real_data=np.zeros(10000)


for i in range(1,10000):
		if s[i]<0.3:
			Real_data[i]=np.random.normal(-5,0.5,1)
		else:
			Real_data[i]=np.random.normal(14,0.5,1)
R=Real_data.reshape(-1,1)
gmm = mixture.GaussianMixture(n_components=2,covariance_type='tied',max_iter=100)
gmm.fit(R)
best_gmm = gmm
y_pred = best_gmm.predict(R)
w_g=best_gmm.weights_
means=best_gmm.means_
covariance=best_gmm.covariances_
print(means)



def _log_multivariate_normal_density_full(X, means, covars,w_g, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples=8
    n_dim = tf.constant(1.0)
    nmix = len(means)
    #log_prob =np.zeros((n_samples, nmix))
    cv1=covars

    #log_prob=tf.Variable([784,0],tf.float64)
    for c, (mu, cv) in enumerate(zip(means, covariance)):
        try:
            cv_chol = tf.cholesky(cv1)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = tf.cholesky(cv1 + min_covar * tf.eye(n_dim))
                                          
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * tf.reduce_sum(tf.log(tf.diag_part(cv_chol)))
        mid=tf.multiply(n_dim , tf.log(2 * np.pi))
        mid1=tf.cast(mid,tf.float64)
        mid2=tf.cast((X-mu),tf.float64)
        cv_sol = tf.transpose(tf.matrix_triangular_solve(cv_chol,tf.transpose(mid2)))
        l = - .5 * tf.add(tf.reduce_sum(cv_sol ** 2, axis=1) ,
                                 tf.add(mid1,cv_log_det))
        l=tf.reshape(l,[-1,8])
        if c==0:
            log_prob=l
            print(log_prob)
        if c!=0:
               log_prob=tf.concat([log_prob,l],axis=0) 
    log_prob=tf.transpose(log_prob)
    lpr=tf.add(log_prob,tf.log(w_g))
    lpr_=tf.reduce_logsumexp(lpr,axis=1)
    lpr_=tf.cast(lpr_,tf.float32)
    lpr_=tf.exp(lpr_)
    return lpr_





class DataDistribution(object):
    def __init__(self):
        self.mu1 = 4
        self.sigma1 = 0.5
	self.mu2=20
	self.sigma2=0.5



    def sample(self, N):
	samples=np.random.choice(Real_data,N)
	
        #samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.001


def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, 20, 'g0'))
    h1 = tf.nn.softplus(linear(h0,4,'g1'))
    h2 = linear(h1, 1, 'g2')
    return h2


def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)


def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))


class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            self.G = generator(self.z, params.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch
            )

        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both

	#self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
	#self.loss_g = tf.reduce_mean(log(self.D2))
	self.s=1-(_log_multivariate_normal_density_full(self.G,means,covariance,w_g))
        self.loss_d = tf.reduce_mean(-log(self.D1) - tf.multiply(log(1 - self.D2),(1-(_log_multivariate_normal_density_full(self.G,means,covariance,w_g))))) 
        self.loss_g = tf.reduce_mean(-tf.multiply(log(self.D2),2*(1-(_log_multivariate_normal_density_full(self.G,means,covariance,w_g)))))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def train(model, data, gen, params):
    anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
	saver = tf.train.Saver()
	


        for step in range(params.num_steps + 1):
            # update discriminator
            x = data.sample(params.batch_size)
            z = gen.sample(params.batch_size)
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, 1)),
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            # update generator
            z = gen.sample(params.batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, 1))
            })
	    #y=_log_multivariate_normal_density_full(means,covariance)
            if step % params.log_every == 0:
                #print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
		print(session.run([model.s],{ model.z: np.reshape(z, (params.batch_size, 1))}))

            if params.anim_path and (step % params.anim_every == 0):
                anim_frames.append(
                    samples(model, session, data, gen.range, params.batch_size)
                )

        if params.anim_path:
            save_animation(anim_frames, params.anim_path, gen.range)
        else:
            samps = samples(model, session, data, gen.range, params.batch_size)
            plot_distributions(samps, gen.range)
	
	#g_new = np.zeros((8, 1))
	for i in range(1,10):
        	z1 = np.random.uniform(-1, 1, size=[8, 1])
		g_new = session.run(model.G, feed_dict={model.z: z1})
		print(g_new)
		

def samples(
    model,
    session,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):
    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(
                    xs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )

    # data distribution
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(
                    zs[batch_size * i:batch_size * (i + 1)],
                    (batch_size, 1)
                )
            }
        )
    g1=g[0:8]
    y1=_log_multivariate_normal_density_full(g1,means,covariance,w_g)
    y=tf.exp(y1)
    #print(session.run(y1))
    print(session.run(y1))
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_animation(anim_frames, anim_path, sample_range):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))

    def init():
        line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return (line_db, line_pd, line_pg, frame_number)

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames))
        )
        db, pd, pg = anim_frames[i]
        line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return (line_db, line_pd, line_pg, frame_number)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])




def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=16), args)
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=10000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    return parser.parse_args()


if __name__ == '__main__':
		main(parse_args())
