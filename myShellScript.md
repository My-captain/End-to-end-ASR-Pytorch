
# 基于ChinaPopSong数据集进行声学模型训练, load参数为resume参数
    - njobs 为Dataloader线程数, 大于0时会出现 killed by signal
    - name参数为tensorboard名称
```shell
# 以中文字符为建模单元
--config ./config/chinaPopSong/asr_hybrid.yaml --njobs 12 --no-pin --name CTCAttChinaPopSong --load ./ckpt/100w_/latest.pth
# 以普通话因素为建模单元
--config ./config/chinaPopSong/asr_hybrid_phoneme.yaml --njobs 0 --no-pin --name CTCAttPhoneme
```


# 基于ChinaPopSong数据集进行语言模型训练
```shell
--config config/chinaPopSong/lm_chinaPopSong.yaml --name chinaPopSongLM --lm --njobs 0
```

# Test
```shell
--config config/chinaPopSong/test_config.yaml --test --njobs 0 --name testChinaPopSong
```
