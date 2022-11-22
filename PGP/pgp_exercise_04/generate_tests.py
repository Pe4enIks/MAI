import os
import random
import argparse

def main(tc=100, 
         n_range=(100, 10000),
         el_range=(-10000, 10000)):
    path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    path += 'tests' + os.sep

    files = os.listdir(path)

    for file in files:
        os.remove('{}{}'.format(path, file))
    
    for idx in range(0, tc):
        with open('{}test{}.txt'.format(path, idx), 'w+') as file:
            n = random.randint(n_range[0], n_range[1])
            file.write('{}\n'.format(n))
            for _ in range(n):
                str = ''
                for j in range(n):
                    val = random.randint(el_range[0], el_range[1])
                    if j != n - 1:
                        str += '{} '.format(val)
                    else:
                        str += '{}\n'.format(val)
                file.write(str)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tc', type=int, required=True)
    parser.add_argument('-nmin', type=int, required=True)
    parser.add_argument('-nmax', type=int, required=True)
    parser.add_argument('-elmin', type=float, required=True)
    parser.add_argument('-elmax', type=float, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    tests_count = args.tc
    n_range = (args.nmin, args.nmax)
    el_range = (args.elmin, args.elmax)

    main(tests_count, n_range, el_range)
