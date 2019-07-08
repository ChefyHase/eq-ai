const Data = require('./data/data.js');
const model = require('./model.js');
const tf = require('@tensorflow/tfjs-node-gpu');

(async ()=> {
  // await model.train({ modelName: 'freq', model: 0, iterations: 100, save: true });
  // await model.train({ modelName: 'gain', model: 1, iterations: 100, save: true });
  await model.train({ modelName: 'q', model: 2, iterations: 10, save: true });
})();
