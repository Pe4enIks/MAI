import numpy as np

def main(name='./debug.txt'):
    with open(name) as file:
        line = file.readline().strip().split()

    n = int(line[0])
    num_els = 2 * n * n

    m1 = list(map(float, line[1:n * n + 1]))
    m2 = list(map(float, line[n * n + 1:2 * n * n + 1]))
    
    new_m1 = np.zeros((n, n))
    new_m2 = np.zeros((n, n))
    new_dot = np.zeros((n, n))

    for num, el in enumerate(m1):
        i, j = int(num % n), int(num // n)
        new_m1[i][j] = el

    for num, el in enumerate(m2):
        i, j = int(num % n), int(num // n)
        new_m2[i][j] = el

    print('init matrix\n', new_m1, '\n')
    print('inverse matrix\n', new_m2, '\n')

    dot = new_m1 @ new_m2
    for i in range(n):
        for j in range(n):
            new_dot[i][j] = round(dot[i][j])

    print('mult matrix\n', new_dot, '\n')


if __name__ == '__main__':
    main()
