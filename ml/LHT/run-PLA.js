var parser = require('./lib/parser').binaryClassificationParser;
var LinearSolver = require('./lib/linear-solver');
var obj = parser('./data/pla/data.dat');
var datas = obj.datas;
var labels = obj.labels;

var linearSolver = new LinearSolver();

function shuffle(o) {
    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x) {}
    return o;
}
var w = [0, 0, 0, 0];
//single times
// var cost = linearSolver.runPLA(w, datas, labels);

//multiple times
var c = 0;
for (var k = 0; k < 2000; k++) {
    var ary = [];
    for (var i = 0; i < datas.length; ary.push(i++));
    ary = shuffle(ary);
    var cc = linearSolver.runPLA(w, datas, labels, ary);
    c += cc
}
console.log(c / 2000)
