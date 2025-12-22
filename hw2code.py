
import numpy as np
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional


def find_best_split(feature_vector: np.ndarray, target_vector: np.ndarray):
    """
    Ищет оптимальный порог для вещественного признака по критерию Джини (в виде качества разбиения):

        Q(R) = - (|Rl|/|R|) * H(Rl) - (|Rr|/|R|) * H(Rr),
        H(R) = 1 - p1^2 - p0^2, где p1/p0 — доли классов 1/0.

    Указания из задания:
    * Пороги, приводящие к пустому левому или правому подмножеству, не рассматриваются.
    * Порог — среднее двух соседних (в отсортированном по признаку векторе) значений.
    * При одинаковом качестве выбираем минимальный порог.
    * Векторизуем (без циклов по объектам/порогам).

    Returns
    -------
    thresholds : np.ndarray
        Отсортированные возможные пороги.
    ginis : np.ndarray
        Значения Q(R) для каждого порога.
    threshold_best : float
        Лучший порог.
    gini_best : float
        Лучшее значение Q(R).
    """
    feature_vector = np.asarray(feature_vector).ravel()
    target_vector = np.asarray(target_vector).ravel().astype(int)

    n = feature_vector.shape[0]
    if n <= 1:
        return np.array([]), np.array([]), None, None

    # Сортируем по признаку
    order = np.argsort(feature_vector, kind="mergesort")  # стабильная сортировка важна для "минимального" порога
    x = feature_vector[order]
    y = target_vector[order]

    # Кандидаты-пороги только между различными соседними значениями
    diff = x[1:] != x[:-1]
    if not np.any(diff):  # константный признак
        return np.array([]), np.array([]), None, None

    thresholds = (x[:-1] + x[1:]) / 2.0
    thresholds = thresholds[diff]

    # Кумулятивные суммы числа "единиц" слева
    y1 = (y == 1).astype(int)
    c1 = np.cumsum(y1)  # c1[i] = #class1 in y[:i+1]
    left_n_all = np.arange(1, n)  # размер левой части для разреза после i-ой позиции (между i и i+1)
    left_n = left_n_all[diff]

    left_ones = c1[:-1][diff]
    left_zeros = left_n - left_ones

    total_ones = c1[-1]
    total_zeros = n - total_ones

    right_n = n - left_n
    right_ones = total_ones - left_ones
    right_zeros = total_zeros - left_zeros

    # Без пустых частей (diff уже исключил края, но на всякий случай)
    valid = (left_n > 0) & (right_n > 0)
    thresholds = thresholds[valid]
    left_n = left_n[valid]
    right_n = right_n[valid]
    left_ones = left_ones[valid]
    right_ones = right_ones[valid]
    left_zeros = left_n - left_ones
    right_zeros = right_n - right_ones

    # H(R) = 1 - p1^2 - p0^2
    # p1 = ones/n, p0 = zeros/n
    pl1 = left_ones / left_n
    pr1 = right_ones / right_n
    Hl = 1.0 - pl1**2 - (1.0 - pl1)**2
    Hr = 1.0 - pr1**2 - (1.0 - pr1)**2

    ginis = - (left_n / n) * Hl - (right_n / n) * Hr

    # Лучшее: максимум ginis. При равенстве — минимальный порог.
    best_idx = np.argmax(ginis)
    gini_best = float(ginis[best_idx])
    threshold_best = float(thresholds[best_idx])

    # На случай нескольких одинаковых максимумов — выбрать минимальный порог
    max_g = ginis[best_idx]
    ties = np.where(np.isclose(ginis, max_g))[0]
    if ties.size > 1:
        tie_thresholds = thresholds[ties]
        min_t = np.min(tie_thresholds)
        threshold_best = float(min_t)
        gini_best = float(max_g)

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Небольшое дерево решений для бинарной классификации с поддержкой
    real/categorical признаков.

    feature_types: список строк длины n_features: "real" или "categorical".
    """
    def __init__(
        self,
        feature_types: List[str],
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
    ):
        if np.any(list(map(lambda x: x not in ("real", "categorical"), feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree: Dict[str, Any] = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    @staticmethod
    def _majority_class(y: np.ndarray) -> int:
        c = Counter(y)
        # Counter.most_common returns list of (label, count)
        return int(c.most_common(1)[0][0])

    def _fit_node(self, sub_X: np.ndarray, sub_y: np.ndarray, node: Dict[str, Any], depth: int = 0):
        # Условия остановки
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = int(sub_y[0])
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = self._majority_class(sub_y)
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = self._majority_class(sub_y)
            return

        best = {
            "feature": None,
            "gini": None,
            "threshold": None,
            "categories_split": None,
            "split_mask": None,
        }

        n_features = sub_X.shape[1]
        for feature in range(n_features):
            ftype = self._feature_types[feature]

            if ftype == "real":
                feature_vector = sub_X[:, feature].astype(float)
                thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
                if threshold is None:
                    continue
                split_mask = feature_vector < threshold

            elif ftype == "categorical":
                # Упорядочиваем категории по доле класса 1 (как в лекции)
                values = sub_X[:, feature]
                counts = Counter(values)
                ones = Counter(values[sub_y == 1])
                # score = P(y=1|cat)
                score = {k: (ones.get(k, 0) / counts[k]) for k in counts}
                sorted_cats = [k for k, _ in sorted(score.items(), key=lambda kv: kv[1])]
                categories_map = {cat: i for i, cat in enumerate(sorted_cats)}
                mapped = np.vectorize(categories_map.get)(values).astype(float)

                thresholds, ginis, threshold, gini = find_best_split(mapped, sub_y)
                if threshold is None:
                    continue
                split_mask = mapped < threshold
                left_cats = [cat for cat, idx in categories_map.items() if idx < threshold]

            else:
                raise ValueError("Unknown feature type")

            # min_samples_leaf constraint
            if self._min_samples_leaf is not None:
                if split_mask.sum() < self._min_samples_leaf or (~split_mask).sum() < self._min_samples_leaf:
                    continue

            if best["gini"] is None or gini > best["gini"]:
                best["feature"] = feature
                best["gini"] = gini
                best["split_mask"] = split_mask
                if ftype == "real":
                    best["threshold"] = threshold
                    best["categories_split"] = None
                else:
                    best["threshold"] = None
                    best["categories_split"] = left_cats

        if best["feature"] is None:
            node["type"] = "terminal"
            node["class"] = self._majority_class(sub_y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = best["feature"]
        if self._feature_types[best["feature"]] == "real":
            node["threshold"] = float(best["threshold"])
        else:
            node["categories_split"] = list(best["categories_split"])

        node["left_child"], node["right_child"] = {}, {}
        mask = best["split_mask"]
        self._fit_node(sub_X[mask], sub_y[mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[~mask], sub_y[~mask], node["right_child"], depth + 1)

    def _predict_node(self, x: np.ndarray, node: Dict[str, Any]):
        # Рекурсивный спуск по дереву
        if node.get("type") == "terminal":
            return node["class"]

        feature = node["feature_split"]
        ftype = self._feature_types[feature]
        if ftype == "real":
            if float(x[feature]) < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        else:  # categorical
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._tree = {}
        self._fit_node(np.asarray(X), np.asarray(y), self._tree, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([self._predict_node(x, self._tree) for x in X])
