var parser = require('./lib/parser').binaryClassificationParser;
var LinearSolver = require('./lib/linear-solver');
var obj = parser('./data/pocket/data.dat');
var datas = obj.datas;
var labels = obj.labels;
var testDataObj = ('./data/pocket/test.dat');
var testDatas = testDataObj.datas;
var testLabels = testDataObj.labels;

var linearSolver = new LinearSolver();
var w = [0, 0, 0, 0];
var erRate = 0;
for (var i = 0; i < 2000; i++) {
    console.log(i)

    var ans = linearSolver.runPocket(w, datas, labels, datas, labels, 100);
    console.log('--------')
    console.log(ans)
    console.log('--------')
    erRate += ans.errorRate
}
console.log(erRate / 2000)
