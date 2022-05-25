# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-05-05 16-17
@file: prepare.py
"""
import concurrent.futures
import json
import os
import pathlib
import shutil
import uuid
from glob import glob

import dill
import librosa.feature
import soundfile as sf

from tqdm import tqdm


def get_filename_wo_extension(path_dir):
    return pathlib.Path(path_dir).stem


def get_song_corpus(meta_json_path):
    meta_corpus = json.load(open(meta_json_path, "r", encoding="utf8"))
    if len(meta_corpus) < 1:
        return None, None
    file_name = get_filename_wo_extension(meta_json_path)
    singer = file_name.split(" - ")[0]
    corpus = [(i["path"], i["text"]) for i in meta_corpus]
    return singer, corpus


def generate_corpus(base_dir):
    whole_corpus = dict()
    for meta in glob(base_dir):
        singer, corpus = get_song_corpus(meta)
        if singer is None:
            continue
        if singer in whole_corpus:
            whole_corpus[singer].extend(corpus)
        else:
            whole_corpus[singer] = corpus

    wav_scp = []
    text = []
    utt2spk = []
    spk2utt = []

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=12)

    for singer, utt_s in whole_corpus.items():
        singer_id = str(uuid.uuid1()).replace("-", "")
        singer_path = f"/media/zliu-elliot/Jarvis/ChinaPopSong/kaldi/wav/{singer_id}/"
        os.mkdir(singer_path)
        singer_utts = []
        for utt in utt_s:
            utt_id = str(uuid.uuid1()).replace("-", "")
            utt_path = f"{singer_path}{singer_id}_{utt_id}.wav"
            wav_scp.append(f"{singer_id}_{utt_id} {utt_path}")
            text.append(f"{singer_id}_{utt_id} {utt[1]}")
            utt2spk.append(f"{singer_id}_{utt_id} {singer_id}")
            singer_utts.append(f"{singer_id}_{utt_id}")
            pool.submit(shutil.copy, f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/{utt[0]}", utt_path)
        spk2utt.append(f"{singer_id} {' '.join(singer_utts)}")
    with open("wav.scp", "w", encoding="utf8") as fp:
        fp.write("\n".join(wav_scp))
    with open("text", "w", encoding="utf8") as fp:
        fp.write("\n".join(text))
    with open("utt2spk", "w", encoding="utf8") as fp:
        fp.write("\n".join(utt2spk))
    with open("spk2utt", "w", encoding="utf8") as fp:
        fp.write("\n".join(spk2utt))


def solve_sample(json_, wav_path, text, base_dir, utt_id, singer_id, token_dict):
    signal, sr = sf.read(wav_path)
    mel = librosa.feature.melspectrogram(signal, sr).T
    dill.dump(mel, open(f"{base_dir}/{utt_id}.dill", mode="wb"))
    json_[utt_id] = {
        "input": [
            {
                "feat": f"{base_dir}/{utt_id}.dill",
                "name": "input1",
                "shape": [
                    mel.shape[0],
                    mel.shape[1]
                ]
            }
        ],
        "output": [
            {
                "name": "target1",
                "shape": [
                    len(text),
                    len(token_dict)
                ],
                "text": text,
                "token": " ".join(text),
                "tokenid": " ".join([str(token_dict[i]) for i in text])
            }
        ],
        "utt2spk": singer_id
    }


def data2json(base_dir):
    whole_corpus = dict()
    for meta in glob(base_dir):
        singer, corpus = get_song_corpus(meta)
        if singer is None:
            continue
        if singer in whole_corpus:
            whole_corpus[singer].extend(corpus)
        else:
            whole_corpus[singer] = corpus

    wav_scp = []
    text = []
    utt2spk = []
    spk2utt = []

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=12)

    token2idx = {'UNK': 0, }
    idx2token = ['UNK']
    all_text = []
    for singer, utt_s in whole_corpus.items():
        # singer_id = str(uuid.uuid1()).replace("-", "")
        # singer_utts = []
        for utt in utt_s:
            all_text.append(utt[1])
    all_text = set("".join(all_text))
    for idx, ch in enumerate(all_text):
        token2idx[ch] = idx + 1
        idx2token.append(ch)
    json.dump(token2idx, open(f"token2idx.json", mode="w", encoding="utf8"), ensure_ascii=False)

    json_ = dict()
    for singer, utt_s in whole_corpus.items():
        singer_id = str(uuid.uuid1()).replace("-", "")
        singer_utts = []
        for utt in utt_s:
            utt_id = str(uuid.uuid1()).replace("-", "")
            # solve_sample(json_, f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/{utt[0]}", utt[1], f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/feature", utt_id, singer_id, token2idx)
            pool.submit(solve_sample, json_, f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/{utt[0]}", utt[1],
                        f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/feature", utt_id, singer_id, token2idx)
    json.dump(json_, open("/media/zliu-elliot/Jarvis/ChinaPopSong/espnet.json", mode="w", encoding="utf8"),
              ensure_ascii=False)


def combine_all_sentence(meta_path, save_path):
    idx = 0
    lm_corpus = open(save_path, "w", encoding="utf8")
    for meta in glob(meta_path):
        idx += 1
        print(idx)
        meta = json.load(open(meta, mode="r", encoding="utf8"))
        for sentence in meta:
            if len(sentence["text"]) > 3:
                lm_corpus.write(sentence["text"])
                lm_corpus.write("\n")
    lm_corpus.close()


def delete_zero_clip(zero_meta_path):
    with open(zero_meta_path, mode="r", encoding="utf8") as fp:
        lines = fp.readlines()
        lines = [line.replace("error: ", "") for line in lines]
        for wav_path in lines:
            file_name = wav_path.replace("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/audio/", "")
            file_name = file_name.split("/")[0]

            meta = json.load(open(f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta_all/{file_name}"))


def generate_all_phoneme(all_sentence_path):
    syllable2phoneme = json.load(open("phone.json", mode="r", encoding="utf8"))

    from xpinyin import Pinyin
    pinyin_encoder = Pinyin()
    with open(all_sentence_path, mode="r", encoding="utf8") as fp:
        lines = fp.readlines()
    all_pinyin = [pinyin_encoder.get_pinyin(line.replace("\n", "")) for line in lines]
    all_phonemes = []
    for line in all_pinyin:
        syllables = []
        try:
            line = line.split("-")
            for syllable in line:
                syllables.append(syllable2phoneme[syllable])
            syllables = " ".join(syllables)
            all_phonemes.append(syllables+"\n")
        except Exception as e:
            print(e)
            print(line)
    with open("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/all_phoneme.txt", mode="w", encoding="utf8") as vocab:
        vocab.write("".join(all_phonemes))
    pass


def transfer_meta_phoneme(base_dir, out_dir):
    syllable2phoneme = json.load(open("phone.json", mode="r", encoding="utf8"))
    from xpinyin import Pinyin
    encoder = Pinyin()
    whole_time = 0
    for meta in glob(base_dir):
        filename = get_filename_wo_extension(meta)
        meta = json.load(open(meta, mode="r", encoding="utf8"))
        transformed_meta = []
        for item in meta:
            phonemes = []
            syllables = encoder.get_pinyin(item["text"]).split("-")
            try:
                for syllable in syllables:
                    phonemes.append(syllable2phoneme[syllable])
                item["text"] = " ".join(phonemes)
                transformed_meta.append(item)
                whole_time += (item["end"] - item["start"])
            except Exception as e:
                print(e)
        if len(transformed_meta) > 0:
            print(filename)
            json.dump(transformed_meta, open(f"{out_dir}/{filename}.json", mode="w", encoding="utf8"), ensure_ascii=False)
    print(f"总时长： {whole_time} \t{out_dir}")




if __name__ == '__main__':
    # data2json(f"/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta/*.json")

    # json_ = json.load(open("./token2idx.json", mode="r", encoding="utf8"))
    # dict_ = [f"{k} {v}\n" for k,v in json_.items()]
    # with open("/media/zliu-elliot/Jarvis/ChinaPopSong/token2idx.txt", mode="w", encoding="utf8") as fp:
    #     fp.write("".join(dict_))

    # delete_zero_clip("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/zero_length.txt")

    # combine_all_sentence("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta_all/*.json",
    #                      "/media/zliu-elliot/Jarvis/ChinaPopSong/origin/all_sentences.txt")

    # generate_all_phoneme("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/all_sentences.txt")
    transfer_meta_phoneme("/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta/dev/*.json", "/media/zliu-elliot/Jarvis/ChinaPopSong/origin/meta_phoneme/dev")
