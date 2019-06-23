const encoder = require('wav-encoder');
const decoder = require('wav-decoder');
const peaking = require('node-peaking');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');
const _ = require('lodash');
const config = require('../config');


class Data {
  constructor(args) {
    this.samplerate = 44100;
    this.shortTimeSamples = 512 // Math.pow(2, 16);

    this.sounds = [];
    this.filterParams = [];
    this.dataSets = [];

    this.index = 0;
    this.batchIndex = 0;
  }

  randomFilterParam() {
    const params = {
      freq: _.random(20, 20000),
      q: _.random(0.1, 24.0, true),
      gain: _.random(-15.0, 15.0, true),
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

    const shortTimeSounds = _.chunk(soundBuffer, this.shortTimeSamples * 2);
    shortTimeSounds.pop();

    return shortTimeSounds;
  }

  async separation() {
    for (let i = 0; i < config.numSamples; i++) {
      let p = path.join(__dirname, 'sounds', i + '.wav');
      let buffers = await this.separateSound(p, 0);
      // random selection
      let index = 0;
      let randIndex = Array(buffers.length - this.shortTimeSamples).fill(0);
      randIndex = randIndex.map((elem) => {
        return index++;
      });
      randIndex = _.shuffle(randIndex).slice(0, config.samplesPerSong);
      let randL = [];
      for (let i = 0; i < randIndex.length; i++) {
        const sliced = buffers.slice(randIndex[i], randIndex[i] + this.shortTimeSamples);
        randL.push(sliced);
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
        const filtered = this.sounds[i].map((buffer) => {
          return peaking.peaking(buffer, filterParam);
        });

        this.sounds[i] = filtered;
        this.filterParams.push(filterParam);
      }
    }
  }

  fft() {
    for (let i = 0; i < this.sounds.length; i++) {
      const tensor = tf.tensor(this.sounds[i]);
      this.sounds[i] = tf.spectral.rfft(tensor).abs().arraySync();
      tensor.dispose();
    }
  }

  makeDataset() {
    for (let n = 0; n < config.trainEpoches; n++) {
      const xBatch = [];
      const labelBatch = [];
      for (let i = 0; i < config.batchSize; i++) {
        xBatch.push([this.sounds[i]]);
        labelBatch.push([
          this.norm(this.filterParams[i].freq, 20, 20000),
          this.norm(this.filterParams[i].gain, -15.0, 15.0),
          this.norm(this.filterParams[i].q, 0.1, 24.0)
        ]);
      }
      this.dataSets.push([xBatch, labelBatch]);
    }
  }

  nextBatch() {
    this.loadDataset(this.index);
    this.index++;
    return {
      xs: tf.tensor(this.dataSets[0]),
      ys: tf.tensor(this.dataSets[1])
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
    this.dataSets = json;
  }

  disposer(tensors) {
    tensors.forEach((elem) => {
      elem.dispose();
    });
  }

  norm(x, min, max) {
    return (x - min) / (max - min);
  }

  invNorm(y, min, max) {
    return y * (max - min) + min
  }
}

module.exports = Data;
