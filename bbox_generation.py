import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import os.path as osp


in_dir = '/home/anjenson/anjenson_ssip/data/{}/images/'
mask_dir = '/home/anjenson/anjenson_ssip/data/ex_masks/'
annot_dir = '/home/anjenson/anjenson_ssip/data/{}/annotations'


def find_neighbours(x, y, mask, visits_mask):
    new_neighbours = []
    if x + 1 < mask.shape[1] and mask[y, x+1] == 255 and visits_mask[y,x+1] == 0:
        new_neighbours.append((y, x+1))
        visits_mask[y, x+1] = 1
    if x - 1 >= 0 and mask[y, x-1] == 255 and visits_mask[y,x-1] == 0:
        new_neighbours.append((y, x-1))
        visits_mask[y, x-1] = 1

    if y - 1 >= 0:
        if mask[y-1, x] == 255 and visits_mask[y-1,x] == 0:
            new_neighbours.append((y-1, x))
            visits_mask[y-1, x] = 1
        if x + 1 < mask.shape[1] and mask[y-1, x+1] == 255 and visits_mask[y-1,x+1] == 0:
            new_neighbours.append((y-1, x+1))
            visits_mask[y-1, x+1] = 1
        if x - 1 >= 0 and mask[y-1, x-1] == 255 and visits_mask[y-1,x-1]==0:
            new_neighbours.append((y-1, x-1))
            visits_mask[y-1, x-1] = 1

    if y + 1 < mask.shape[0]:
        if mask[y+1, x] == 255 and visits_mask[y+1,x] == 0:
            new_neighbours.append((y+1, x))
            visits_mask[y+1, x] = 1
        if x + 1 < mask.shape[1] and mask[y+1, x+1] == 255 and visits_mask[y+1,x+1] == 0:
            new_neighbours.append((y+1, x+1))
            visits_mask[y+1, x+1] = 1
        if x - 1 >= 0 and mask[y+1, x-1] == 255 and visits_mask[y+1,x-1] == 0:
            new_neighbours.append((y+1, x-1))
            visits_mask[y+1, x-1] = 1

    return new_neighbours, visits_mask


def bfs(image, visits_mask, seed_point):
    min_x, max_x = image.shape[1], 0
    min_y, max_y = image.shape[0], 0
    queue = [seed_point]

    while len(queue) > 0:
        (y, x), queue = queue[0], queue[1:]

        new_neighbours, visits_mask = find_neighbours(x, y, image, visits_mask)

        queue += new_neighbours

        min_x = x if min_x > x else min_x
        min_y = y if min_y > y else min_y
        max_x = x if max_x < x else max_x
        max_y = y if max_y < y else max_y

    return min_x, min_y, max_x, max_y, visits_mask


def process_image(image_path, set_mode):
    image_name = osp.basename(image_path)
    im = cv2.imread(osp.join(mask_dir, image_name), cv2.IMREAD_GRAYSCALE)
    im[im >= 100] = 255
    im[im < 100] = 0
    im_id = osp.splitext(osp.basename(image_path))[0]

    visits_mask = np.zeros(im.shape, np.uint8)

    with open(osp.join(annot_dir.format(set_mode), im_id + '.txt'), 'w') as annot_file:
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if im[i,j] == 255 and visits_mask[i,j] == 0:
                    min_x, min_y, max_x, max_y, visits_mask = bfs(im, visits_mask, (i, j))
                    if min(max_y - min_y, max_x - min_x) > 3:
                        annot_file.write('ex {} {} {} {}\n'.format(min_x, min_y, max_x, max_y))


def process_directory(seed_dir, set_mode):
    images = [osp.join(seed_dir, f) for f in os.listdir(seed_dir)]
    for image in images:
        print 'Processing', image
        process_image(image, set_mode)


if __name__ == '__main__':
    process_directory(in_dir.format('train'), 'train')
    process_directory(in_dir.format('test'), 'test')

