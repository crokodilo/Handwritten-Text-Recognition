import os
import numpy as np

if __name__ == '__main__':
    # Create train valid and test folders
    data_path= '../datasets/iam/'

    if not os.path.exists(os.path.join(data_path, 'imgs')):

        os.mkdir(os.path.join(data_path, 'imgs'))
        os.system(f'tar -xf {data_path}/formsA-D.tgz --directory {data_path}/imgs/')
        os.system(f'tar -xf {data_path}/formsE-H.tgz --directory {data_path}/imgs/')
        os.system(f'tar -xf {data_path}/formsI-Z.tgz --directory {data_path}/imgs/')

        import random
        import shutil

        data_path = os.path.join(data_path, 'imgs')

        all_files = os.listdir(data_path)

        all_png_files = [x for x in all_files if x.endswith('.png')]

        print(set(all_files) - set(all_png_files))

        random.shuffle(all_png_files)

        split = random.choices(['train', 'valid', 'test'], [0.85, 0.075, 0.075], k = len(all_png_files))

        os.mkdir(os.path.join(data_path, 'train'))
        os.mkdir(os.path.join(data_path, 'valid'))
        os.mkdir(os.path.join(data_path, 'test'))

        total_path = lambda file_name: os.path.join(data_path, file_name)

        for file_name, mode in zip(all_png_files, split):

            src_path = total_path(file_name)
            dst_path = os.path.join(data_path, mode, file_name)

            shutil.move(src_path, dst_path)

        for s in ['train', 'valid', 'test']:
            print(s, int(len(os.listdir(os.path.join(data_path, f'{s}')))))

    else:
        for s in ['train', 'valid', 'test']:
            print(s, int(len(os.listdir(os.path.join(data_path, f'imgs/{s}')))))