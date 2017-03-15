import numpy as np

x = np.array([
    [19, 9],
    [15, 7],
    [7, 2],
    [17, 6]
])
y = np.array([1, 1, 2, 2])

x1 = np.array([
    x[0],
    x[1],
    x[2],
    x[3],
])
x2 = np.array([
    x[1],
    x[0],
    x[3],
    x[2],
])

x1x2 = x1 - x2
normx1x2 = np.linalg.norm(x1x2, axis=1)
print('x1x2\n%s\nnormx1x2\n%s\n' % (x1x2, normx1x2))
x1x3 = np.array([
    x1[0] - x1[2],
    x1[0] - x1[3],
    x1[1] - x1[2],
    x1[1] - x1[3],
    x1[2] - x1[0],
    x1[2] - x1[1],
    x1[3] - x1[0],
    x1[3] - x1[1],
])
normx1x3 = np.linalg.norm(x1x3, axis=1)
expx1x3 = np.exp(1 - normx1x3)

li = {}
Li = {}

li[0] = normx1x2[0] + np.log(expx1x3[0] + expx1x3[1] + expx1x3[2] + expx1x3[3])
li[1] = normx1x2[1] + np.log(expx1x3[0] + expx1x3[1] + expx1x3[2] + expx1x3[3])
li[2] = normx1x2[2] + np.log(expx1x3[4] + expx1x3[5] + expx1x3[6] + expx1x3[7])
li[3] = normx1x2[3] + np.log(expx1x3[4] + expx1x3[5] + expx1x3[6] + expx1x3[7])
print('Li\n%s\n' % li)

diffe = {}

diffe[1] = (1 / 4)
