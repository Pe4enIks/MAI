import sys
import imageio
import os


def main():
    images_path = sys.argv[1] + sys.argv[2] + os.sep

    filenames = os.listdir(images_path)

    inds = [int(file.split('_')[1][:-4]) for file in filenames]
    min_ind = min(inds)
    max_ind = max(inds)

    filenames_range = range(min_ind, max_ind + 1)
    filenames = [images_path + f'img_{ind}.png' for ind in filenames_range]

    gif_path = f'{sys.argv[3]}{os.sep}{sys.argv[2]}.gif'
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == '__main__':
    main()
