# -*- coding: utf-8 -*-
"""
feasibility.py -- аналитические тесты выполнимости (без имитации).

Эти тесты формализуют критерии выполнимости из раздела 2.1 и служат
эталоном (точным либо достаточным) для проверки результатов имитационного
моделирования в SimSo. Все функции работают с наборами задач в формате
[(C, T, D), ...] (мс) и не требуют установленного SimSo.

Реализованы:
  * EDF на одном ядре: точный тест U <= 1 (implicit), тест плотности и
    функции спроса (constrained);
  * Fixed Priority (RM/DM): граница загрузки Liu-Layland, гиперболическая
    граница (Bini), точный тест по времени отклика (RTA, рекуррентность из
    раздела 1.5);
  * глобальные достаточные тесты: GFB (Goossens-Funk-Baruah) для G-EDF.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Task = Tuple[float, int, int]  # (C, T, D)


# --------------------------------------------------------------------------- #
#  Вспомогательные величины                                                   #
# --------------------------------------------------------------------------- #
def utilization(tasks: List[Task]) -> float:
    return sum(C / T for C, T, _ in tasks)


def density(tasks: List[Task]) -> float:
    return sum(C / min(T, D) for C, T, D in tasks)


# --------------------------------------------------------------------------- #
#  Одноядерный EDF                                                            #
# --------------------------------------------------------------------------- #
def edf_uniproc_implicit(tasks: List[Task]) -> bool:
    """Точный тест EDF для implicit-deadline (D=T): U <= 1."""
    return utilization(tasks) <= 1.0 + 1e-9


def edf_uniproc_density(tasks: List[Task]) -> bool:
    """Достаточный тест EDF для constrained-deadline: sum C_i/min(T_i,D_i) <= 1."""
    return density(tasks) <= 1.0 + 1e-9


def _dbf(task: Task, t: float) -> float:
    """Функция спроса dbf_i(t) одной задачи на интервале длины t."""
    C, T, D = task
    if t < D:
        return 0.0
    return (math.floor((t - D) / T) + 1) * C


def edf_uniproc_exact(tasks: List[Task], horizon: int = None) -> bool:
    """Точный тест EDF (processor demand criterion, Baruah et al., 1990):
    для всех t: sum_i dbf_i(t) <= t.

    Достаточно проверить t в множестве дедлайнов до границы (La-bound). Здесь
    горизонт ограничивается гиперпериодом (для ограниченных периодов корпуса)."""
    if utilization(tasks) > 1.0 + 1e-9:
        return False
    if horizon is None:
        horizon = _hyperperiod([T for _, T, _ in tasks])
    # Контрольные точки -- абсолютные дедлайны заданий до горизонта.
    points = set()
    for C, T, D in tasks:
        k = 0
        while D + k * T <= horizon:
            points.add(D + k * T)
            k += 1
    for t in sorted(points):
        demand = sum(_dbf(task, t) for task in tasks)
        if demand > t + 1e-9:
            return False
    return True


# --------------------------------------------------------------------------- #
#  Одноядерный Fixed Priority (RM / DM)                                        #
# --------------------------------------------------------------------------- #
def ll_bound(n: int) -> float:
    """Граница загрузки Liu-Layland для RM: n (2^(1/n) - 1)."""
    return n * (2.0 ** (1.0 / n) - 1.0)


def rm_ll_test(tasks: List[Task]) -> bool:
    """Достаточный тест RM по границе Liu-Layland (implicit-deadline)."""
    n = len(tasks)
    return utilization(tasks) <= ll_bound(n) + 1e-9


def rm_hyperbolic_test(tasks: List[Task]) -> bool:
    """Гиперболическая граница (Bini, Buttazzo, Buttazzo, 2003):
    prod (U_i + 1) <= 2. Точнее границы Liu-Layland."""
    prod = 1.0
    for C, T, _ in tasks:
        prod *= (C / T + 1.0)
    return prod <= 2.0 + 1e-9


def response_time(task: Task, higher: List[Task]) -> float:
    """Время отклика задачи методом RTA (рекуррентность раздела 1.5):
        R^(0) = C_i,
        R^(k+1) = C_i + sum_{j in hp(i)} ceil(R^(k)/T_j) * C_j.
    higher -- задачи с более высоким приоритетом. Возвращает inf при
    расходимости (R > D)."""
    C_i, T_i, D_i = task
    R = C_i
    while True:
        interference = sum(math.ceil(R / Tj) * Cj for Cj, Tj, _ in higher)
        R_new = C_i + interference
        if R_new > D_i:
            return float("inf")
        if abs(R_new - R) < 1e-9:
            return R_new
        R = R_new


def fp_rta_test(tasks: List[Task], priority: str = "DM") -> bool:
    """Точный тест выполнимости FP по времени отклика.

    priority: 'RM' -- по возрастанию периода, 'DM' -- по возрастанию
    относительного дедлайна. Возвращает True, если для всех задач R_i <= D_i."""
    if priority == "RM":
        order = sorted(tasks, key=lambda x: x[1])
    elif priority == "DM":
        order = sorted(tasks, key=lambda x: x[2])
    else:
        raise ValueError("priority должно быть 'RM' или 'DM'")
    for i, task in enumerate(order):
        higher = order[:i]
        if response_time(task, higher) > task[2] + 1e-9:
            return False
    return True


# --------------------------------------------------------------------------- #
#  Глобальные достаточные тесты                                               #
# --------------------------------------------------------------------------- #
def gedf_gfb_test(tasks: List[Task], m: int) -> bool:
    """Достаточный тест G-EDF (Goossens, Funk, Baruah, 2003) для
    implicit-deadline: U_sum <= m - (m - 1) * U_max."""
    U = utilization(tasks)
    U_max = max(C / T for C, T, _ in tasks)
    return U <= m - (m - 1) * U_max + 1e-9


# --------------------------------------------------------------------------- #
#  Утилиты                                                                     #
# --------------------------------------------------------------------------- #
def _hyperperiod(periods: List[int]) -> int:
    h = 1
    for p in periods:
        h = h * p // math.gcd(h, p)
    return h


if __name__ == "__main__":
    # Небольшая самопроверка на известном наборе.
    ts = [(1, 4, 4), (2, 6, 6), (3, 8, 8)]   # U = 0.25+0.333+0.375 = 0.958
    print("U =", round(utilization(ts), 4))
    print("EDF implicit (U<=1):", edf_uniproc_implicit(ts))
    print("EDF exact (PDC):    ", edf_uniproc_exact(ts))
    print("RM Liu-Layland:     ", rm_ll_test(ts))
    print("RM hyperbolic:      ", rm_hyperbolic_test(ts))
    print("FP RTA (RM):        ", fp_rta_test(ts, "RM"))
    print("G-EDF GFB (m=2):    ", gedf_gfb_test(ts, 2))
