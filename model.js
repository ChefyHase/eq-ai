const tf = require('@tensorflow/tfjs-node-gpu');
const Data = require('./data/data.js');
const config = require('./config.js');

class Model {
  constructor(args) {
    this.data = new Data();
    this.model;
  }

  build() {
    const droprate = 0.0;
    const input = tf.input({ shape: [2048] });
    const reshape = tf.layers.reshape({ targetShape: [64, 32, 1] }).apply(input);

    const conv1 = tf.layers.conv2d({ filters: 128, kernelSize: 10, strides: 2, padding: 'same' }).apply(reshape);
    const activ1 = tf.layers.reLU().apply(conv1);
    const norm1 = tf.layers.batchNormalization().apply(activ1);
    const pooling1 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(norm1);
    const dropout1 = tf.layers.dropout({ rate: droprate }).apply(pooling1);

    const conv2 = tf.layers.conv2d({ filters: 64, kernelSize: 5, strides: 2, padding: 'same' }).apply(dropout1);
    const activ2 = tf.layers.reLU().apply(conv2);
    const norm2 = tf.layers.batchNormalization().apply(activ2);
    const pooling2 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(norm2);
    const dropout2 = tf.layers.dropout({ rate: droprate }).apply(pooling2);

    const flatten = tf.layers.flatten().apply(dropout2);

    const dense1 = tf.layers.dense({ units: 64 }).apply(flatten);
    const activDense1 = tf.layers.reLU().apply(dense1);

    const dense2 = tf.layers.dense({ units: 25 }).apply(activDense1);
    const activDense2 = tf.layers.reLU().apply(dense2);

    const dense3 = tf.layers.dense({ units: 10 }).apply(activDense2);
    const activDense3 = tf.layers.reLU().apply(dense3);

    const denseOutput = tf.layers.dense({ units: 4 }).apply(activDense3);
    const activOutput = tf.layers.reLU().apply(denseOutput);

    const model = tf.model({ inputs: input, outputs: activOutput });
    model.summary();
    this.model = model;
  }

  async train() {
    this.build();
    console.log('model build: done');

    const optimizer = tf.train.adam(0.0001);
    this.model.compile({ optimizer: optimizer, loss: 'meanSquaredError', metrics: ['accuracy'] });

    for (let i = 0; i < 1; ++i) {
      const { xs, ys } = this.data.nextBatch();

      const h = await this.model.fit(xs, ys, {
        batchSize: 100,
        epochs: 100,
        shuffle: true,
        validationSplit: 0.3
      });

      console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
      await this.model.save('file://./mastering-ai');

      xs.dispose();
      ys.dispose();

      // let buffer = await data.separateSound(__dirname + '/test.wav');
      // let xs = data.fft(tf.tensor(buffer));
    }
  }

  async predict(wav) {
    const model = await tf.loadLayersModel('file://./mastering-ai/model.json');
    const predictBatch = await this.data.predictBatch(wav);
    const prediction = model.predict(predictBatch);

    const mean = prediction.mean(0).dataSync();
    const freq = this.data.invNorm(mean[0], 20, 20000);
    const q = this.data.invNorm(mean[1], 0, 5.0);
    const gain = this.data.invNorm(mean[2], -15.0, 15.0);
    const bw = this.data.invNorm(mean[3], 0, 5.0);

    console.log(freq, q, gain, bw);
  }
}

module.exports = new Model();
