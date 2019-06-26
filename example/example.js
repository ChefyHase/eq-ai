const eq = require('../index.js');

(async() => {
  let res = await eq('./Mixdown.wav', 1);
  console.log(res);
})()
