const eq = require('../index.js');

(async() => {
  let res = await eq('./test.wav', 5);
  console.log(res);
})()
