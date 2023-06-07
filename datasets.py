import numpy as np
import torch
import pandas as pd
import re, os
from PIL import Image


class HandWrittenDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder = '../datasets/iam/', split = 'train', transform = None, tokenizer = None, number_of_lines = 1):
        assert number_of_lines > 0
        assert split in {'train', 'valid', 'test'}
        
        self.path = data_folder
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        
        with open(os.path.join(self.path, 'lines.txt'), 'r') as f:
            lines = f.readlines()

        self.description = lines[:23]
        data = lines[23:]

        rows = []
        for line in data:
            temp = line.split()
            row = temp[:8]
            row.extend([' '.join(temp[8:])])
            rows.append(row)

        cols = ['id', 'result', 'graylevel', 'components', 'x', 'y', 'w', 'h', 'message']

        df = pd.DataFrame(rows, columns=cols)
        for col in ['components', 'graylevel', 'x', 'y', 'w', 'h']:
            df[col] = df[col].astype('int64')

        def clean_message(m):
            m = m.replace('|-|', ' - ')
            m = m.replace('|*|', ' * ')
            temp0 = m.replace('.|"', '"|.')
            temp1 = re.sub(r'"\|([\w\d\s\|-]+)\|"', r'"\1"', temp0)
            temp2 = re.sub(r'([^\'\(])\|([\w\d\"\#\(])', r'\1 \2', temp1)
            temp3 = re.sub(r'([^\'\(\"])\|([\w\d\"\#\(])', r'\1 \2', temp2)
            temp4 = re.sub(r'^"\s', r'"', temp3)
            return temp4.replace('|', '').replace(' ". ', '". ')

        df['clean_message'] = df['message'].apply(clean_message)
        
        self.df = df
        self.max_target_length = 30*number_of_lines
        self.length = number_of_lines
        
        file_list = os.listdir(os.path.join(self.path, 'imgs', self.split))
        
        self.img_names = [x.replace('.png', '') for x in file_list if x.endswith('.png')]
        
    def getchunk(self, idx):
        
        img_name = self.img_names[idx]
        sample_df = self.df[self.df['id'].str.contains(img_name + '-')]
        image = Image.open(os.path.join(self.path, 'imgs', self.split, f'{img_name}.png')).convert("RGB")
        delta = 20
        
        start = np.random.choice(range(max(len(sample_df) - self.length + 1, 1)))

        array = np.asarray(image)
        xmin = min(sample_df['x'])
        xmax = max(sample_df['x']) + max(sample_df['w'])

        ymin = sample_df['y'].iloc[start]

        end = min(start + min(self.length, len(sample_df)), len(sample_df)) - 1

        ymax = sample_df['y'].iloc[end] + sample_df['h'].iloc[end]
        
        l, w, _ = array.shape
        a = max(ymin-delta,0)
        b = min(ymax+delta,l-1)
        c = max(xmin-delta,0)
        d = min(xmax+delta,w-1)
        
        img = Image.fromarray(array[a:b, c:d, :])
        
        return {'image': img, 'text': ' '.join(sample_df['clean_message'].iloc[start:end+1])}

    
    def __getitem__(self, idx):
        
        chunk = self.getchunk(idx)
        img, txt = chunk['image'], chunk['text']
        
        img = self.transform(img)
        
        caption_ids = self.tokenizer(txt, padding='max_length', max_length = 30).input_ids
        caplen = caption_ids.index(self.tokenizer.pad_token_id)
        
        return  img, torch.LongTensor(caption_ids), torch.LongTensor([caplen])
    
    def __len__(self):
        return len(self.img_names)