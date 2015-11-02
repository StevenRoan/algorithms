var gen1dX = function (interval, number) {
    var ini = interval[0];
    var ret = [];
    for (var i = 0; i < number; i++) {
        ret.push(ini + (interval[1] - ini) * Math.random());
    }
    return ret;
}
var genNoiseYByX = function (X, flipProb) {
    var ret = [];
    for (var i = 0; i < X.length; i++) {
        var noise = Math.random();
        var x = X[i] > 0 ? 1 : -1;
        x = noise > flipProb ? x : -1 * x;
        ret.push(x);
    }
    return ret;
}


var getCost = function (X, Y, stump, direction) {
    var cost = 0;
    for (var i = 0; i < X.length; i++) {
        var x = X[i] >= stump ? 1 * direction : -1 * direction;
        if (x !== Y[i]) {
            cost++;
        }
    }
    return cost / X.length;
};
var stumpAlg = function (Xin, Yin) {
    var minCost, minStump, minRay;
    // console.log('stumpAlg')
    for (var i = 0; i < Xin.length; i++) {
        var costPR = getCost(Xin, Yin, Xin[i], 1);
        var costNR = getCost(Xin, Yin, Xin[i], -1);
        // console.log(costPR, costNR, minCost, minRay)
        if (costPR < minCost || minCost == undefined) {
            minCost = costPR;
            minStump = Xin[i];
            minRay = 1;
        }

        if (costNR < minCost || minCost == undefined) {
            minCost = costNR;
            minStump = Xin[i];
            minRay = -1;
        }
    }
    // console.log('---------')
    return {
        minCost: minCost,
        minStump: minStump,
        minRay: minRay
    }
}
var train1d = function (Xin, Yin, Xout, Yout) {
    var res = stumpAlg(Xin, Yin);
    res.eoutCost = getCost(Xout, Yout, res.minStump, res.minRay);
    res.theoryEout = 0.5 + 0.3 * (Math.abs(res.minStump) - 1);
    return res;
}

var doMath = function () {
    var X = gen1dX([-1, 1], 20);
    var Y = genNoiseYByX(X, 0.2);
    var Xout = gen1dX([-1, 1], 20);
    var Yout = genNoiseYByX(X, 0.2);
    var res = train1d(X, Y, Xout, Yout);
    return res;
}

var einsum = 0,
    eoutSum = 0;
for (var i = 0; i < 5000; i++) {
    var res = doMath();
    einsum += res.minCost;
    eoutSum += res.theoryEout;
}
console.log(einsum / 5000);
console.log(eoutSum / 5000);

exports.doStump1d = stumpAlg;
