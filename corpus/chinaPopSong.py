# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-05-06 21-55
@file: chinaPopSong.py
"""
import json
import math
import shutil
from glob import glob
from os.path import join
from pathlib import Path

import torch.utils.data
from torch.utils.data import Dataset
from tqdm import tqdm


class ChinaPopAudioDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.audio_path_prefix = "/media/zliu-elliot/Jarvis/ChinaPopSong/origin/"
        self.tokenizer = tokenizer

        clip_list = []
        text = []
        for s in split:
            split_list = list(Path(join(path, s)).rglob("*.json"))
            for meta_ in split_list:
                json_ = json.load(open(meta_, "r", encoding="utf8"))
                if len(json_) < 1:
                    continue
                for info in json_:
                    clip_list.append(info["path"])
                    text.append(info["text"])
        text = [tokenizer.encode(txt) for txt in text]
        # Sort dataset by text length
        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(clip_list, text), reverse=not ascending, key=lambda x:len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list)-self.bucket_size, index)
            return [(self.audio_path_prefix+f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
        else:
            file_path = self.audio_path_prefix + self.file_list[index]
            text_tokens = self.text[index]
            return file_path, text_tokens

    def __len__(self):
        return len(self.file_list)


def split_meta(base_dir, train, dev, test):
    meta_json = glob(f"{base_dir}/*.json")
    size = len(meta_json)
    train, dev = math.floor(size * train), math.floor(size * dev)
    train, dev, test = torch.utils.data.random_split(meta_json, [train, dev, size-train-dev])
    for idx in train:
        shutil.copyfile(idx, f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta/train/{idx.split('/')[-1]}")
    for idx in dev:
        shutil.copyfile(idx, f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta/dev/{idx.split('/')[-1]}")
    for idx in test:
        shutil.copyfile(idx, f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta/test/{idx.split('/')[-1]}")


class ChinaPopTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        # 训练时encode
        self.encode_on_fly = False
        read_txt_src = []
        # List all wave files
        file_list, all_sent = [], []
        for s in split:
            with open(join(path, s), 'r') as f:
                all_sent += f.readlines()
        # Encode text
        if self.encode_on_fly:
            self.tokenizer = tokenizer
            self.text = all_sent
        else:
            self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
        del all_sent

        # sort dataset by file size
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text)-self.bucket_size, index)
            if self.encode_on_fly:
                for i in range(index, index+self.bucket_size):
                    if type(self.text[i]) is str:
                        self.text[i] = self.tokenizer.encode(self.text[i])
            # Return a bucket
            return self.text[index:index+self.bucket_size]
        else:
            if self.encode_on_fly and type(self.text[index]) is str:
                self.text[index] = self.tokenizer.encode(self.text[index])
            return self.text[index]

    def __len__(self):
        return len(self.text)


def split_sentence(base_dir, train, dev, test):
    with open(f"{base_dir}/all_sentences.txt", "r", encoding="utf8") as fp:
        lines = fp.readlines()
    size = len(lines)
    train, dev = math.floor(size * train), math.floor(size * dev)
    train, dev, test = torch.utils.data.random_split(lines, [train, dev, len(lines) - train - dev])
    if len(train) > 0:
        with open(f"{base_dir}/sentence/train.txt", mode="w", encoding="utf8") as fp:
            fp.write("".join(train))
    if len(test) > 0:
        with open(f"{base_dir}/sentence/test.txt", mode="w", encoding="utf8") as fp:
            fp.write("".join(test))
    if len(dev) > 0:
        with open(f"{base_dir}/sentence/dev.txt", mode="w", encoding="utf8") as fp:
            fp.write("".join(dev))



if __name__ == '__main__':
    split_meta("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta_all", 0.8, 0.2, 0)
    # split_sentence("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/", 0.8, 0.2, 0)
