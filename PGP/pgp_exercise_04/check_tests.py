import os
import numpy as np

def main(p1, p2, p3, eps):
    files = os.listdir(p1)

    autotest_file = open(p3, 'w+')

    for k, file in enumerate(files):
        m1, m2 = [], []

        with open('{}{}{}'.format(p1, os.sep, file)) as f1:
            data1 = f1.readlines()

        with open('{}{}{}'.format(p2, os.sep, file)) as f2:
            data2 = f2.readlines()

        n = int(data1[0].strip())
        for i in range(1, n + 1):
            arr1 = list(map(float, data1[i].strip().split()))
            arr2 = list(map(float, data2[i - 1].strip().split()))
            m1.append(arr1)
            m2.append(arr2)
        
        m1, m2 = np.array(m1), np.array(m2)
        diff = m1 @ m2 - np.eye(n)

        mae = 0.0
        for i in range(n):
            for j in range(n):
                mae += abs(diff[i][j])
        mae /= n * n
        
        if mae < eps:
            res_str = "test {} was passed, mae {}\n".format(k, mae)
            autotest_file.write(res_str)
        else:
            res_str = "test {} wasn't passed, mae {}\n".format(k, mae)
            autotest_file.write(res_str)
    
    autotest_file.close()


if __name__ == '__main__':
    p1, p2, p3, eps = './tests', './results', './autotest.txt', 0.01
    main(p1, p2, p3, eps)
