class NNM(object):
    #layers is a dictionary of layers with key being the type of the layer and value an array of layers
    def __init__(self, layers):
        self.layers = layers
        self.epoch = 0

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.layers["output"].set_target(self.y)
        self.epoch += 1
        print("----- Start " + str(self.epoch) + " epoch")
        self.fp()
        self.bp()
        self.clear()
        print("----- Finish " + str(self.epoch) + " epoch")
        self.show()

    def show(self):
        pass

    def fp(self):
        #First, go through convolutional layer
        c = self.layers["convo"].f(self.x)
        p = c.then(self.layers["pooling"].f)
        h = p.then(self.layers["hidden"].f)
        o = h.then(self.layers["output"].f)
        print(o.value)


    def bp(self):
        #if you don't want to pass in value, you don't need promises
        o = self.layers["output"].b(None)

        h = o.then(self.layers["hidden"].b)

        p = h.then(self.layers["pooling"].b)
        c = p.then(self.layers["convo"].b)


    def clear(self):
        for key, value in self.layers.items():
            # self.layers[key].output = None
            value.output = None



    def add_layer(self, layer):
        self.layers.append(layer)

    def get_layers(self):
        return self.layers

    def set_layer(self, index, layer):
        self.layers[index] = layer


