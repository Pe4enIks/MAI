import random


def main(name, nc, h, w):
    np = []

    for _ in range(nc):
        npj = random.randint(10000, 524288)
        if npj % 2 != 0:
            npj -= 1
        lst = [npj]
        for _ in range(npj):
            x, y = random.randint(0, w), random.randint(0, h)
            lst.append(x)
            lst.append(y)
        np.append(lst)

    with open(name + '.txt', 'w+') as file:
        file.write(f'../{name}.data\n')
        file.write(f'../classified_{name}.data\n')
        file.write(f'{nc}\n')
        for i in range(nc):
            s = ' '.join(list(map(str, np[i])))
            file.write(f'{s}\n')


if __name__ == '__main__':
    name = 'big'
    nc, h, w = 32, 5000, 8000

    main(name, nc, h, w)
