const eq = require('../index.js');

(async() => {
  const res = await eq('./test.wav', 5);
  console.log(res);
})()
