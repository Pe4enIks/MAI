def main(name, nc, np):
    with open(name + '.txt', 'w+') as file:
        file.write(f'../{name}.data\n')
        file.write(f'../classified_{name}.data\n')
        file.write(f'{nc}\n')
        for i in range(nc):
            s = ' '.join(list(map(str, np[i])))
            file.write(f'{s}\n')


if __name__ == '__main__':
    name = 'test'
    nc = 3
    cls1 = [4, 0, 0, 1920, 1393, 1700, 1100, 420, 820]
    cls2 = [4, 700, 1120, 1280, 1050, 1300, 600, 1000, 1000]
    cls3 = [3, 849, 821, 959, 875, 825, 769]
    cls = [cls1, cls2, cls3]

    main(name, nc, cls)
