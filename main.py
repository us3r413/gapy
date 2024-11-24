# -*- coding: utf-8 -*-

import csv
import json
import os

import yaml

from display import plot_init
from ga import GA
from location import Location
from utility import rand

# 載入設定檔
with open("config.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

dcf = config['data']
if dcf['from_file']:  # 從檔案讀取資料
    fmt = dcf['file']['format']
    path = dcf['file']['path']
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        if fmt == 'csv':
            reader = csv.reader(f)
            data = []
            for row in reader:
                data.append(Location(row))
            del reader
        elif fmt == 'json':
            data = list(map(Location, json.load(f)))
        else:
            raise ValueError(f"不支援的檔案格式: {fmt}")
    del fmt, path
else:  # 產生隨機資料
    size = dcf['random']['size']
    low, high = dcf['random']['min'], dcf['random']['max']
    data = [Location(rand.uniform(low, high, 2)) for _ in range(size)]
    del size, low, high


# 設定人口數
mode = config['ga']['population']['mode']
if mode == 'value':  # 人口數為固定值
    config['ga']['population'] = config['ga']['population']['value']
elif mode == 'ratio':  # 人口數為資料點數的比例
    config['ga']['population'] = len(data) * config['ga']['population']['value']

plot_init(config['plot'])

del f, dcf, mode


if __name__ == '__main__':
    ga = GA(data, **config['ga'])
    ga.run()
    ga.output_result()
