import json
import multiprocessing as mp
import os
import shutil
from collections import defaultdict
from functools import partial

import numpy as np
from numpy import double
from tqdm import tqdm

from .display import plot_route, plot_summary
from .location import Location, Route
from .utility import AdvancedJSONEncoder, rand

bits = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)


class GA:
    def __init__(
        self,
        locations: list[Location],
        generations: int = None,
        population: int = None,
        k: int = None,
        crossover_rate: float = None,
        mutation_rate: float = None,
        **kwargs
    ) -> None:
        """
        基因演算法 (Genetic Algorithm, GA)

        Args:
            locations (list[Location]): 所有城市的座標
            generations (int):          要演化幾代，預設為 100 代
            population (int):           群體的大小，預設為城市數量的 100 倍
            k (int):                    基因要切成幾段，預設為 2 段
            crossover_rate (float):     表現最好的前幾名才能夠交配，值必須在 [0, 1] 之間，預設為 0.8 (80%)
            mutation_rate (float):      基因突變率，值必須在 [0, 1] 之間，預設為 0.1 (10%)
        """
        self._locations = locations
        self._generations = generations or 100
        self._population = population or len(locations) * 100
        self._k = min(k or 2, len(locations))  # 每段基因至少有一個城市
        self._crossover_rate = crossover_rate or 0.8
        self._mutation_rate = mutation_rate or 0.1

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._pool_size = mp.cpu_count()  # 使用 CPU 核心數量

        self.best: Route = None  # 歷史最佳路徑
        self._history: list[dict] = []  # 每一代的歷史紀錄

        # 初始化人口
        self._individuals = [Route.random(locations) for _ in range(self._population)]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def history(self):
        """
        輸出最佳路徑
        """
        return self._history

    def load_history(self) -> None:
        """
        載入歷史紀錄
        """
        folder = self.folder or "results"
        with open(f'{folder}/history.json', 'r', encoding='utf-8') as f:
            self._history = json.load(f)

    def output_result(self):
        """
        輸出結果
        """
        # 刪除舊的圖片
        folder = self.folder or "results"
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

        with open(f'{folder}/history.json', 'w', encoding='utf-8') as f:
            json.dump(self._history, f, ensure_ascii=False, cls=AdvancedJSONEncoder, indent=2)

        summary = defaultdict(list)

        for d in self._history:
            for k, v in d['summary'].items():
                summary[k].append(v)

        plot_summary(f'{folder}/summary.png', summary)

        args = [
            (f"{folder}/{h['generation']:03d}.png", h['generation'], h['best_now'], h['best_all'])
            for h in self._history
        ]

        with mp.Pool(self._pool_size) as pool:
            pool.starmap(plot_route, tqdm(args, desc='畫圖中'))

    def run(self) -> None:
        """
        開始演化
        """
        self.best = None
        self._history = []

        # self.pool = mp.Pool(self._pool_size)

        with mp.Pool(self._pool_size) as pool:
            self.pool = pool
            for gen in tqdm(range(self._generations), desc='演化進度'):  # 世代數
                # 演化 (好像在初始化時就已經做了)

                # 紀錄結果
                self._best_now = min(self._individuals)
                if self.best is None or self._best_now.distance < self.best.distance:
                    self.best = self._best_now

                result = np.array([i.distance for i in self._individuals])

                self._history.append({
                    'generation': gen,
                    "best_now": self._best_now,
                    "best_all": self.best,
                    "summary": {
                        "mean": np.mean(result),
                        "std": np.std(result),
                        "min": np.min(result),
                        "q1": np.percentile(result, 25),
                        "median": np.median(result),
                        "q3": np.percentile(result, 75),
                        "max": np.max(result),
                    },
                })

                del result

                # 最後一代不用交配
                if gen == self._generations - 1:
                    continue

                # 交配
                self.crossover()

        # self.pool.shutdown(wait=True)

    def crossover(self) -> None:
        """
        交配
        """
        # 預處理可以交配的基因
        idvs = sorted(self._individuals)[:int(self._population * self._crossover_rate)]
        self._individuals.clear()  # 殺死上一代

        # print(idvs)
        dis = np.array([i.distance for i in idvs])
        idxs = np.arange(len(idvs))

        p = (self._population - idxs) / np.array(dis, dtype=double)  # 距離愈短，機率愈高
        p = p / np.sum(p)  # 機率總和為 1

        crossover = partial(Route.crossover, k=self._k, mutation_rate=self._mutation_rate)

        fs = [(idvs[f[0]], idvs[f[1]]) for f in rand.choice(idxs, (self._population, 2), p=p)]
        fs = tqdm(fs, desc='繁衍中', leave=False, position=1)

        self._individuals = self.pool.starmap(crossover, fs)
