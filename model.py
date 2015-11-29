import numpy as np
import theano
import theano.tensor as T
theano.config.floatX = 'float32'
from sklearn.base import ClassifierMixin
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import pickle
# import NN
from imp import reload
# reload(NN)
import Plots
reload(Plots)

def relu(x):
    return T.switch(x < 0, 0, x)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]

class Meta_ConvNet(ClassifierMixin):
    """Pool Layer of a convolutional network """

    def __init__(self, n_epochs=1, batch_size=1,
                 learning_rate=0.1, momentum=0.0,
                 filters=[(50, 3, 11, 11)], image_shape=[], poolsize=(2, 2),
                 denoising=0.0,
                 dropout=0.0,
                 tied_weights=True,
                 loss="MSE",
                 pooling=False,
                 #activation=T.tanh):
                 #activation=T.nnet.sigmoid):
                 activation=relu):
                 
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.filters = filters
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.n_epochs = n_epochs
        self.denoising = denoising  #0.0 means no noise, 1.0 means complete noise
        self.dropout = dropout
        self.tied_weights = tied_weights
        self.loss = loss
        self.pooling = pooling
        self.activation = activation

    #---------------------------------------------------------
    def get_convout_vol(self, in_shape, f_shape):
        convmap_s = in_shape[2] - f_shape[2] + 1
        return (f_shape[0], convmap_s, convmap_s)

    #---------------------------------------------------------
    def get_poolout_vol(self, in_shape, poolsize):
        pooledmap_s = (in_shape[2] - poolsize[0])/2.0 + 1
        return (in_shape[1], pooledmap_s, pooledmap_s)

    #---------------------------------------------------------
    # fit the model to the data
    def fit(self, X, y=None):
        X = X.astype(np.float32)
        
        self.costs = []
        N = X.shape[0]
        print("Fit %d examples (%d, %d, %d, %d) over %d epochs\n" %
              (N, X.shape[0], X.shape[1], X.shape[2], X.shape[3],
               self.n_epochs))
        count = 0
        num_plotted=0
        for epoch in range(self.n_epochs):
            perm = np.random.permutation(N)
            #print("Epoch: %d\n" % epoch)
            for i in range(0, N, self.batch_size):
                if i + self.batch_size <= N:
                    # plotting updates
                    if count%50 == 0:
                        self.plotStuff(X, self.Ws, num_plotted)
                        #self.reconstruct(y)
                        num_plotted += 1
                    count += 1

                    # continue with training
                    rowindexes = perm[i:i+self.batch_size]
                    minibatch_x = X[rowindexes, :, :, :]
                                       
                    cost, out, g, outbefore = self.train(minibatch_x)
                    print("\n(%d/%d) %d/%d cost: %d" %
                          (epoch+1, self.n_epochs, i, N, cost))
                    print("output (min,max): (%f, %f)" %
                          (np.min(out), np.max(out)))
                    print("weights (min,max): (%f, %f)" %
                          (np.min(self.Ws[0].get_value()), np.max(self.Ws[0].get_value())))
                    print("grads (min,max): (%f, %f)" %
                          (np.min(g), np.max(g)))
                    print("outputshape:", out.shape)
                    print("outbefore:", np.max(outbefore))

                    f = open('costs.txt', 'a')
                    f.write("\n(%d/%d) %d/%d cost: %f \n" %
                            (epoch+1, self.n_epochs, i, N, cost))
                    f.write("output (min,max): (%f, %f) \n" %
                            (np.min(out), np.max(out)))
                    f.write("weights (min,max): (%f, %f) \n" %
                            (np.min(self.Ws[0].get_value()), np.max(self.Ws[0].get_value())))
                    f.write("grads (min,max): (%f, %f) \n" %
                            (np.min(g), np.max(g)))
                    f.close()
   
                    self.costs.append(cost)
            Plots.plot_costs(self.costs)
            print("cost: ", cost)
            f = open('costs.txt', 'a')
            f.write('epoch %d) cost: %f \n' % (epoch, cost))
            f.close()


        # do some predictions:
        perm = np.random.permutation(N)
        rowindexes = perm[:self.batch_size]
        minibatch_x = X[rowindexes, :, :, :]
        
        predictions = self.predict(minibatch_x)
        self.samples = minibatch_x
        self.preds = predictions
        self.W1 = self.Ws[0].get_value()
        if len(self.Ws) > 1:
            self.W2 = self.Ws[1].get_value()

        
        pickle.dump(self.Ws, open("weights.p", "wb"))
        return self

    #---------------------------------------------------------
    def reconstruct(self, Y):
        print(Y.shape)
        Y = Y.reshape((Y.shape[0], 1, 28, 28))
        Y = Y[:self.batch_size, :]
        Y = Y.astype(np.float32)                  
        Yout = self.predict(Y)[0]
        print(Y.shape)
        print(Yout.shape)
        Plots.plot_predictions_grey(Y, Yout, 99999999, [self.batch_size, 1,28,28])
        return

    #---------------------------------------------------------
    def get_corrupted_block(self, input, corruption_level):
        size = (input.shape[1], )
        print("size", size)
        noise = self.theano_rng.binomial(size=size, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX)
        return noise.dimshuffle('x', 0, 'x', 'x') * input
        
    #---------------------------------------------------------
    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
        
    #---------------------------------------------------------
    def plotStuff(self, X, Ws, num_plotted):

        Plots.plot_filters(Ws[0].get_value(), self.image_shape[1], num_plotted, title="layer1")
        if len(Ws) > 1:
            Plots.plot_filters(Ws[1].get_value(), self.image_shape[1], num_plotted, title="layer1")
        samples = X[:self.batch_size, :, :, :]
        
        predictions, conv_volume0, conv_volume1, conv_volume2, conv_volume3, conv_volume4, deconv_volume = self.predict(samples)

        Plots.plot_predictions_grid(samples, predictions, num_plotted, self.image_shape)
        
        Plots.plot_volume(conv_volume0, num_plotted, title="layer0 ")
        Plots.plot_volume(conv_volume1, num_plotted, title="layer1 ")
        Plots.plot_volume(conv_volume2, num_plotted, title="layer2 ")
        Plots.plot_volume(conv_volume3, num_plotted, title="layer3 ")
        Plots.plot_volume(conv_volume4, num_plotted, title="layer4")

        Plots.plot_costs(self.costs)
        return 
        

        
    #---------------------------------------------------------
    def generate(self):
        size = (self.batch_size, 20, 18, 18)
        block = np.random.randn(*size)
        block = block.astype(np.float32)
        out = self.gen(block)
        print(out.shape)
        Plots.plot_testimg(out)
        return out
        
    #---------------------------------------------------------
    def setup(self):
        self.rng = np.random.RandomState(1234)
        # for denoising
        self.theano_rng = T.shared_randomstreams.RandomStreams(self.rng.randint(2 ** 30))

        x = T.tensor4('x')
        input = x
        if self.denoising > 0.0:
            input = self.get_corrupted_input(x, self.denoising)

        # bookkeeping
        conv_vols = []
        pool_vols = []
        Ws = []
        bs = []
        deConv_bs = []
        i = 1
        params = []
        in_shape = self.image_shape
        c=0
        conv_volume0 = x
        conv_volume1 = x
        conv_volume2 = x
        conv_volume3 = x
        conv_volume4 = x
        for f in self.filters:
            print("----------------Conv layer  -----------------------")
            fan_in = f[1] * f[2] * f[3]
            fan_out = 0
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            
            W = theano.shared(np.asarray(self.rng.uniform(low=-4.0 * W_bound,
                                                          high=4.0 * W_bound,
                                                          size=f),
                                         dtype=theano.config.floatX),
                              borrow=True, name="Conv_W")
            W = theano.shared(floatX(np.random.randn(*f) * 0.01))
            params.append(W)
            Ws.append(W)
            conv_out = conv.conv2d(
                input=input,
                filters=W,
                border_mode="valid"
            )
            b_values = np.zeros((f[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True, name="Conv_b")
            params.append(b)
            
            output = self.activation(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
            
            conv_map_s = self.get_convout_vol(in_shape, f)
            conv_vols.append(conv_map_s)
            print("conv out vol:", conv_map_s)

            # setup the input into the next layer
            in_shape = (self.batch_size,
                        conv_map_s[0], conv_map_s[1], conv_map_s[2])

            input = output

            # dropout
            if self.dropout > 0.0:
                input = self.get_corrupted_block(output, self.dropout)
            
            if c==0:
                conv_volume1 = input
            elif c==1:
                conv_volume2 = input
            c += 1


            if self.pooling:
                print("--------------Pool layer---------------------------")
                pooled_out = downsample.max_pool_2d(
                    input=input,
                    ds=(2, 2),
                    ignore_border=True
                )
                output = pooled_out
                pool_map_s = self.get_poolout_vol(in_shape, (2, 2))
                pool_vols.append(pool_map_s)
                print("pool out 1 vol:", pool_map_s)
                # setup the input into the next layer
                in_shape = (self.batch_size, f[0],
                            int(pool_map_s[1]), int(pool_map_s[2]))
                input = output

        # for generating "fantasized" data, TODO
        halfpass = input
        input = halfpass
        
        i = len(self.filters) - 1
        for f in reversed(self.filters):
            if self.pooling:
                print("------------------------Up sample layer 2-----------------")
                inp2 = input
                pool_map_s = pool_vols[i]
                shp = (self.batch_size, int(pool_map_s[0]), int(pool_map_s[1] * 2),
                       int(pool_map_s[2] * 2))
                upsample = T.zeros(shp, dtype=inp2.dtype)
                upsample = T.set_subtensor(upsample[:, :, ::2, ::2], inp2)

                # setup the input into the next layer
                in_shape = (self.batch_size,
                            int(pool_map_s[0]), int(pool_map_s[1]),
                            int(pool_map_s[2]))
                print(shp)
                input = upsample

            print("--------------------------DeConv layer 2------------------")
            if self.tied_weights:
                W = Ws[i]
                defilter = W.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
            else:
                fan_in = f[1] * f[2] * f[3]
                fan_out = 1
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                defilter = theano.shared(np.asarray(self.rng.uniform(low=-4.0 * W_bound,
                                                                    high=4.0 * W_bound,
                                                                    size=(f[1], f[0], f[2], f[3])),
                                                dtype=theano.config.floatX),
                                        borrow=True, name="Conv_W")
                params.append(defilter)
            deconv_out = conv.conv2d(
                input=input,
                filters=defilter,
                border_mode="full"
            )
   
            i -= 1

            #print("-------------------------------------------------")
            deConv_b_values = np.zeros((self.batch_size,), dtype=theano.config.floatX)
            deConv_b = theano.shared(value=deConv_b_values, borrow=True, name="DeConv_b")
            params.append(deConv_b)
            output = self.activation(deconv_out +
                                     deConv_b.dimshuffle(0, 'x', 'x', 'x'))
            deconv_volume = output
            input = output
            if i==0:
                conv_volume3 = output
            else:
                conv_volume4 = output

        output = input.flatten(2)
        outbeforeclip = output
        flat_x = x.flatten(2)

        # MSE
        if self.loss == "MSE":
            L = T.sum((flat_x - output)**2, axis=1)

        # Cross-entropy
        elif self.loss == "CE":
            output = T.clip(output, 10**(-6), 1-10**(-6))
            L = -T.sum(flat_x * T.log(output) +
                       (1-flat_x) * T.log(1-output), axis=1)

        L1 = sum(abs(W).sum() for W in Ws)
        L2 = sum((W ** 2).sum() for W in Ws)
        reg = 0.001 * L2 + 0.0001 * L1
        self.cost = L.mean()
        outs = [output]
        outs = outs + [conv_volume0, conv_volume1, conv_volume2, conv_volume3, conv_volume4]
        outs = outs + [deconv_volume]
        
        grads = T.grad(cost=self.cost, wrt=params)
        updates = []
        lr=0.001
        rho=0.9
        epsilon=1e-6
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
       
        gs = T.grad(self.cost, Ws[0])
        
        self.predict = theano.function([x], outs)
        self.train = theano.function(inputs=[x], outputs=[self.cost, output, gs, outbeforeclip],
                                updates=updates)
        self.gen = theano.function([halfpass], output)
        self.Ws = Ws

