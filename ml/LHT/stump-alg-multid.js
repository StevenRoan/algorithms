var parser = require('./lib/parser').binaryClassificationParser;
var obj = parser('./data/stump/train.dat');
var testobj = parser('./data/stump/test.dat');
var datas = obj.datas;
var labels = obj.labels;

var doStump1d = require('./stump-alg-1d').doStump1d;

var getColumnX = function (datas) {
    var Xi = [];
    for (var i = 0; i < datas[0].length; i++) {
        var x = [];
        for (var j = 0; j < datas.length; j++) {
            x.push(datas[j][i]);
        }
        Xi.push(x);
    }
    return Xi;
}
var colX = getColumnX(obj.datas);
var getDecisionVector = function (colX, Y) {
    var decisionVector = [];
    for (var i = 0; i < colX.length; i++) {
        var res = doStump1d(colX[i], Y);
        decisionVector.push({
            stump: res.minStump,
            ray: res.minRay
        });
    }
    return decisionVector;
}
var decisionVec = getDecisionVector(colX, obj.labels);
console.log(decisionVec)

var getCostOf = function (X, Y, decisionVector) {
    var cost = 0;
    for (var i = 0; i < X.length; i++) {
        var pJudge = 0;
        for (var j = 0; j < X[i].length; j++) {
            if (X[i][j] * decisionVector[j].ray > decisionVector[j].stump* decisionVector[j].ray) {
                pJudge++;
            }
        }
        var y = (pJudge > X[i].length / 2) ? 1 : -1;
        if (y !== Y[i]) {
            cost++;
        }
    }
    return cost;
};
var costin = getCostOf(obj.datas, obj.labels, decisionVec);
var costout = getCostOf(testobj.datas, testobj.labels, decisionVec);
console.log(costin / obj.datas.length);
console.log(costout / testobj.datas.length);
