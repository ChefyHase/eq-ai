const eq = require('../index.js');

(async() => {
  const res = await eq('./test.wav', 3);
  console.log(res);
})()
