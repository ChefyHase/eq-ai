const encoder = require('wav-encoder');
const decoder = require('wav-decoder');
const peaking = require('node-peaking');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const _ = require('lodash');
const config = require('../config');


class Data {
  constructor(args) {
    this.samplerate = 44100;
    this.shortTimeSamples = 1024; // Math.pow(2, 16);

    this.sounds = [];
    this.filteredSounds = [];
    this.filterParams = [];
    this.dataSets = [];

    this.index = 0;
    this.batchIndex = 0;
  }

  randomFilterParam() {
    const params = {
      freq: this.lin2log(_.random(0, 1, true)),
      q: _.random(0.1, 24.0),
      gain: _.random(-24.0, 24.0),
      samplerate: this.samplerate
    }
    return params;
  }

  async separateSound(sound, chunnel = 0) {
    let soundBuffer;
    if (typeof sound === 'object') {
      soundBuffer = sound;
    }
    else {
      const soundFilaPeth = path.resolve(sound);
      soundBuffer = await decoder.decode(fs.readFileSync(soundFilaPeth));
    }
    soundBuffer = soundBuffer.channelData[chunnel];

    const shortTimeSounds = _.chunk(soundBuffer, this.shortTimeSamples);
    shortTimeSounds.pop();

    return shortTimeSounds;
  }

  async separation() {
    for (let i = 0; i < config.numSamples; i++) {
      let p = path.join(__dirname, 'sounds', i + '.wav');
      let buffers = await this.separateSound(p, 0);
      // random selection
      let index = 0;
      let randIndex = Array(buffers.length).fill(0);
      randIndex = randIndex.map((elem) => {
        return index++;
      });
      randIndex = _.shuffle(randIndex).slice(0, config.samplesPerSong);
      let randL = [];
      for (let i = 0; i < randIndex.length; i++) {
        randL.push(buffers[randIndex[i]]);
      }

      if (config.varbose) console.log(i + ' / ' + config.numSamples);

      this.sounds.push(...randL);
    }
  }

  async applyFilter(data = null, param = null) {
    if (data && param) {
      const filterParam = param;
      const separated = await this.separateSound(data);
      const filtered = [];
      for (let buffer of separated) {
        filtered.push(...peaking.peaking(buffer, filterParam))
      }
      return filtered;
    }
    else {
      for (let i = 0; i < this.sounds.length; i++) {
        const filterParam = this.randomFilterParam();
        const filtered = peaking.peaking(this.sounds[i], filterParam);
        this.filteredSounds.push(filtered);
        this.filterParams.push(filterParam);
      }
    }
  }

  makeDataset() {
    for (let n = 0; n < config.trainEpoches; n++) {
      const xBatch = [];
      const raw = [];
      const labelBatch = [];
      for (let i = 0; i < config.batchSize; i++) {
        xBatch.push(...Array(...this.filteredSounds[this.batchIndex]));
        raw.push(...this.sounds[this.batchIndex]);
        labelBatch.push([
          this.log2lin(this.filterParams[this.batchIndex].freq),
          this.norm(this.filterParams[this.batchIndex].gain, -24.0, 24.0),
          this.norm(this.filterParams[this.batchIndex].q, 0.1, 24.0)
        ]);
        this.batchIndex++;
      }
      this.dataSets.push([xBatch, raw, labelBatch]);
    }
  }

  nextBatch(index) {
    if (index) this.index = index;
    this.loadDataset(this.index);
    this.index++;
    return {
      xs: tf.tensor(this.dataSets[0], null, 'float32'),
      raw: this.dataSets[1],
      ys: tf.tensor(this.dataSets[2], null, 'float32')
    }
  }

  async predictBatch(wav) {
    const buffer = await this.separateSound(wav);
    const ts = tf.tensor(buffer);
    return ts;
  }

  loadDataset(index) {
    const filePath = config.dataSetPath + index + '.json';
    let json = JSON.parse(fs.readFileSync(filePath));
    const xs = _.chunk(json[0], this.shortTimeSamples);
    const raw = _.chunk(json[1], this.shortTimeSamples);
    this.dataSets = [xs, raw, json[2]];
  }

  disposer(tensors) {
    tensors.forEach((elem) => {
      elem.dispose();
    });
  }

  norm(x, min, max) {
    return (x - min) / (max - min);
  }

  normTens(xs) {
    return tf.div(xs.sub(xs.min()), xs.max().sub(xs.min()));
  }

  invNorm(y, min, max) {
    if (y >= 1) y = 1.0;
    return y * (max - min) + min
  }

  lin2log(x) {
    if (x >= 1) x = 1.0;
    let b = Math.log(20000/20) / (1 - 0);
    let a = 20 / Math.exp(b * 0);
    return a * Math.exp(b * x);
  }

  log2lin(x) {
    let b = Math.log(20000/20) / (1 - 0);
    let a = 20 / Math.exp(b * 0);
    return Math.log(x / a) / b;
  }

  valSplits(x, rate) {
    const train = [];
    const val = [];

    tf.unstack(x).forEach((tensor, index) => {
      if (index < config.batchSize - config.batchSize * rate) train.push(tensor.arraySync());
      else val.push(tensor.arraySync());
    });
    return [tf.tensor(train), tf.tensor(val)];
  }
}

module.exports = Data;
