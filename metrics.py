# -*- coding: utf-8 -*-
"""
metrics.py -- агрегирование метрик и статистический анализ (раздел 2.1).

Функции принимают результаты прогонов (словарь, возвращаемый runner.run_one)
и вычисляют сводные показатели качества планирования, формализованные в
разделе 2.1: коэффициент выполнимости, предельную (breakdown) загрузку,
взвешенную выполнимость, времена отклика, джиттер, долю пропусков дедлайнов,
накладные расходы (вытеснения, миграции). Дополнительно реализованы
статистические процедуры сравнения алгоритмов.

Зависит только от numpy и scipy; SimSo не требуется.
"""

from __future__ import annotations

from typing import List, Dict, Any, Sequence

import numpy as np

try:
    from scipy import stats as _sps
    _HAS_SCIPY = True
except Exception:                       # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
#  Базовые метрики качества планирования                                      #
# --------------------------------------------------------------------------- #
def schedulability_ratio(results: Sequence[Dict[str, Any]]) -> float:
    """S_A(U) = доля наборов, признанных выполнимыми (формула 2.1)."""
    if not results:
        return float("nan")
    feasible = sum(1 for r in results if r["schedulable"])
    return feasible / len(results)


def deadline_miss_ratio(results: Sequence[Dict[str, Any]]) -> float:
    """DMR = (сумма пропусков) / (сумма заданий) по выборке (формула 2.4)."""
    misses = sum(r["deadline_misses"] for r in results)
    jobs = sum(r["jobs_total"] for r in results)
    return misses / jobs if jobs else float("nan")


def breakdown_utilization(u_values: Sequence[float],
                          ratios: Sequence[float],
                          threshold: float = 0.99) -> float:
    """Эмпирическая предельная загрузка U* (формула 2.2): наибольшее значение
    нормированной загрузки U/m, при котором коэффициент выполнимости ещё не
    ниже threshold. Если порог не достигнут нигде, возвращается NaN."""
    best = float("nan")
    for u, s in sorted(zip(u_values, ratios)):
        if s >= threshold:
            best = u
        else:
            break
    return best


def weighted_schedulability(u_values: Sequence[float],
                            ratios: Sequence[float]) -> float:
    """Взвешенная выполнимость W_A (Bastoni, Brandenburg, Anderson, 2010):
        W = sum_u (U * S_A(U)) / sum_u U.
    Сжимает кривую выполнимости в одно число в [0,1], придавая больший вес
    высоким загрузкам."""
    u = np.asarray(u_values, dtype=float)
    s = np.asarray(ratios, dtype=float)
    denom = u.sum()
    return float((u * s).sum() / denom) if denom > 0 else float("nan")


def mean_wcrt(results: Sequence[Dict[str, Any]]) -> float:
    """Среднее по наборам худшее время отклика (нормируется отдельно)."""
    vals = [max(r["wcrt_per_task"]) for r in results if r["wcrt_per_task"]]
    return float(np.mean(vals)) if vals else float("nan")


def mean_jitter(results: Sequence[Dict[str, Any]]) -> float:
    """Средний джиттер отклика по задачам (формула 2.3)."""
    vals = [j for r in results for j in r["jitter_per_task"]]
    return float(np.mean(vals)) if vals else float("nan")


def mean_overhead(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Средние накладные расходы: вытеснения и миграции на набор (формула 2.5)."""
    if not results:
        return {"preemptions": float("nan"), "migrations": float("nan")}
    pre = float(np.mean([r["preemptions"] for r in results]))
    mig = float(np.mean([r["migrations"] for r in results]))
    return {"preemptions": pre, "migrations": mig}


# --------------------------------------------------------------------------- #
#  Описательная статистика и доверительные интервалы                          #
# --------------------------------------------------------------------------- #
def median_iqr(values: Sequence[float]) -> Dict[str, float]:
    """Медиана и межквартильный размах (устойчивые к выбросам оценки)."""
    a = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1)}


def bootstrap_ci(values: Sequence[float], stat=np.mean,
                 n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Бутстреп-доверительный интервал для статистики stat (по умолчанию --
    среднее) уровня 1-alpha методом перцентилей."""
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(a, size=a.size, replace=True)
        boots[b] = stat(sample)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(stat(a)), "lo": float(lo), "hi": float(hi)}


# --------------------------------------------------------------------------- #
#  Статистическое сравнение алгоритмов                                         #
# --------------------------------------------------------------------------- #
def wilcoxon_test(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Парный критерий Уилкоксона (знаково-ранговый) для двух алгоритмов на
    одних и тех же наборах. Возвращает статистику и p-value."""
    if not _HAS_SCIPY:
        raise RuntimeError("Требуется scipy")
    stat, p = _sps.wilcoxon(x, y)
    return {"statistic": float(stat), "pvalue": float(p)}


def friedman_test(*groups: Sequence[float]) -> Dict[str, float]:
    """Критерий Фридмана для сравнения k>=3 алгоритмов на общих наборах
    (Demšar, 2006). Возвращает статистику хи-квадрат и p-value."""
    if not _HAS_SCIPY:
        raise RuntimeError("Требуется scipy")
    stat, p = _sps.friedmanchisquare(*groups)
    return {"statistic": float(stat), "pvalue": float(p)}


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05
                    ) -> List[bool]:
    """Поправка Холма-Бонферрони на множественные сравнения (post-hoc после
    критерия Фридмана). Возвращает список «отвергнута ли H0» для каждой
    гипотезы в исходном порядке."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if pvalues[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


if __name__ == "__main__":
    # Самопроверка агрегаций на синтетических результатах.
    np.random.seed(0)
    fake = []
    for k in range(100):
        sched = k < 80   # 80% выполнимы
        fake.append({
            "schedulable": sched,
            "deadline_misses": 0 if sched else 3,
            "jobs_total": 50,
            "wcrt_per_task": [12.0, 30.0, 7.5],
            "jitter_per_task": [1.0, 2.5, 0.5],
            "preemptions": 40 + k % 5,
            "migrations": (0 if sched else 10),
        })
    print("S_A      =", round(schedulability_ratio(fake), 3))
    print("DMR      =", round(deadline_miss_ratio(fake), 4))
    us = [0.5, 0.6, 0.7, 0.8, 0.9]
    ss = [1.0, 1.0, 0.995, 0.7, 0.2]
    print("U* (0.99)=", breakdown_utilization(us, ss))
    print("W        =", round(weighted_schedulability(us, ss), 4))
    print("mean WCRT=", round(mean_wcrt(fake), 3))
    print("overhead =", mean_overhead(fake))
    print("median/iqr WCRT(all)=",
          median_iqr([w for r in fake for w in r["wcrt_per_task"]]))
    ci = bootstrap_ci([w for r in fake for w in r["wcrt_per_task"]],
                      n_boot=2000)
    print("bootstrap CI mean WCRT=",
          {k: round(v, 3) for k, v in ci.items()})
    if _HAS_SCIPY:
        a = np.random.normal(10, 1, 30)
        b = a + np.random.normal(0.5, 1, 30)
        c = a + np.random.normal(1.0, 1, 30)
        print("Friedman =",
              {k: round(v, 4) for k, v in friedman_test(a, b, c).items()})
        print("Holm     =", holm_bonferroni([0.001, 0.04, 0.20]))
