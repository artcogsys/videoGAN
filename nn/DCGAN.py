import chainer
import chainer.functions as F
import chainer.links as L

class Discriminator(chainer.Chain):

    def __init__(self, ch=128):
        super(Discriminator, self).__init__()
        self.ch = ch
        self.up_sample_dim = ((4, 4, 2))
        self.out_size = self.xp.prod(self.up_sample_dim)

        with self.init_scope():

            w = chainer.initializers.HeNormal(0.02)
            self.c1 = L.Convolution3D(3, ch//8, ksize=4, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution3D(ch//8, ch//4, ksize=4, stride=2, pad=1, initialW=w)
            self.c3 = L.Convolution3D(ch//4, ch//2, ksize=4, stride=2, pad=1, initialW=w)
            self.c4 = L.Convolution3D(ch//2, ch, ksize=4, stride=2, pad=1, initialW=w)
            self.c5 = L.Convolution3D(ch, 1, ksize=3, stride=1, pad=1, initialW=w)
            self.l1 = L.Linear(self.out_size, 1, initialW=w)

            self.ln2 = L.LayerNormalization() # all have layer norm except first and last layer
            self.ln3 = L.LayerNormalization()
            self.ln4 = L.LayerNormalization()

    def forward(self, x):
        h1 = F.leaky_relu(self.c1(x))
        h2 = F.leaky_relu(self.c2(h1))
        h2 = self.reshape_and_layer_norm(h2, self.ln2)
        h3 = F.leaky_relu(self.c3(h2))
        h3 = self.reshape_and_layer_norm(h3, self.ln3)
        h4 = F.leaky_relu(self.c4(h3))
        h4 = self.reshape_and_layer_norm(h4, self.ln4)
        h5 = F.leaky_relu(self.c5(h4)) # No batchnorm

        return self.l1(h5)

    def reshape_and_layer_norm(self, h, lnorm):
        s = h.shape
        h = F.reshape(h, [s[0], -1]) # flatten to allow computation of stats on second axis
        h = lnorm(h)
        h = F.reshape(h, s)

        return h

    def add_noise(self, h, sigma=0.1):
        randn = self.xp.random.randn(*h.shape)

        return h + sigma * randn


class Generator(chainer.Chain):

    def __init__(self, ch=128, latent_dim=100):
        super(Generator, self).__init__()
        self.ch = ch
        self.latent_dim = latent_dim
        self.up_sample_dim = ((ch, 4, 4, 2))
        self.out_size = self.xp.prod(self.up_sample_dim)

        with self.init_scope():

            w = chainer.initializers.Normal(0.02)
            self.l1 = L.Linear(self.latent_dim, self.out_size, initialW=w)
            self.dc1 = L.Deconvolution3D(ch, ch//2, ksize=4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution3D(ch//2, ch//4, ksize=4, stride=2, pad=1, initialW=w)
            self.dc3 = L.Deconvolution3D(ch//4, ch//8, ksize=4, stride=2, pad=1, initialW=w)
            self.dc4 = L.Deconvolution3D(ch//8, 3, ksize=4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(self.out_size)
            self.bn2 = L.BatchNormalization(ch//2)
            self.bn3 = L.BatchNormalization(ch//4)
            self.bn4 = L.BatchNormalization(ch//8)

    def forward(self, latent_z):

        h1 = F.leaky_relu(self.bn1(self.l1(latent_z)))
        h1 = F.reshape(h1, (len(latent_z),) + self.up_sample_dim)
        h2 = F.leaky_relu(self.bn2(self.dc1(h1)))
        h3 = F.leaky_relu(self.bn3(self.dc2(h2)))
        h4 = F.leaky_relu(self.bn4(self.dc3(h3)))
        x  = F.tanh(self.dc4(h4))

        return x

    def sample_hidden(self, batch_size):
        xp = self.xp
        z_layer = xp.random.normal(size=(batch_size, self.latent_dim, 1, 1)).astype(xp.float32)

        return z_layer

