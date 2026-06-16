#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taskset_generator.py
====================

Генератор синтетических наборов периодических задач реального времени для
сравнительного анализа алгоритмов планирования в многоядерных ОСРВ.

Реализованные методы:
  * UUniFast (Bini, Buttazzo, 2005)  -- несмещённое распределение суммарной
    загрузки U между n задачами для одноядерного случая;
  * UUniFast-Discard (Davis, Burns, 2009) -- расширение UUniFast на случай
    m процессоров: наборы, в которых хотя бы одна задача имеет U_i > 1
    (физически невыполнимо на одном ядре), отбрасываются и генерируются
    заново;
  * логравномерная генерация периодов (Emberson, Stafford, Davis, 2010).

Модель задачи: tau_i = (C_i, T_i, D_i),
  C_i -- наихудшее время выполнения (WCET, мс),
  T_i -- период активации (мс),
  D_i -- относительный дедлайн (мс):
         implicit   -> D_i = T_i,
         constrained-> C_i <= D_i <= T_i.

Все случайные величины порождаются из numpy.random.Generator с фиксированным
seed, что обеспечивает полную воспроизводимость корпуса.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np


# --------------------------------------------------------------------------- #
#  Базовые генераторы                                                         #
# --------------------------------------------------------------------------- #
def uunifast(n: int, total_u: float, rng: np.random.Generator) -> List[float]:
    """
    Алгоритм UUniFast (Bini & Buttazzo, 2005).

    Возвращает n коэффициентов использования u_i, в сумме дающих total_u,
    равномерно распределённых по симплексу {u_i > 0, sum u_i = total_u}.
    """
    utilizations = []
    sum_u = total_u
    for i in range(1, n):
        # next_sum = sum_u * U^(1/(n-i)), где U ~ Uniform(0,1)
        next_sum = sum_u * (rng.random() ** (1.0 / (n - i)))
        utilizations.append(sum_u - next_sum)
        sum_u = next_sum
    utilizations.append(sum_u)
    return utilizations


def uunifast_discard(n: int, total_u: float, rng: np.random.Generator,
                     max_tries: int = 10000) -> List[float]:
    """
    UUniFast-Discard (Davis & Burns, 2009).

    Повторяет UUniFast до тех пор, пока все u_i <= 1. Возвращает первый
    допустимый вектор. Если за max_tries попыток допустимый набор не найден
    (характерно для total_u, близкого к m при малом n), бросает RuntimeError.
    """
    for _ in range(max_tries):
        u = uunifast(n, total_u, rng)
        if all(0.0 < ui <= 1.0 for ui in u):
            return u
    raise RuntimeError(
        f"UUniFast-Discard: не удалось получить допустимый набор за "
        f"{max_tries} попыток (n={n}, U={total_u:.3f}).")


def divisor_period_set(base_period: int, t_min: float,
                       t_max: float) -> List[int]:
    """
    Множество делителей base_period, попадающих в [t_min, t_max].

    Если все периоды задач выбираются из делителей одного базового периода,
    то гиперпериод набора (НОК периодов) гарантированно делит base_period и
    тем самым ограничен сверху. Это делает имитационное моделирование
    вычислительно осуществимым (длительность прогона = один гиперпериод).
    Платой является частичная гармоничность периодов, что обсуждается в
    разделе ограничений методики.
    """
    divs = [d for d in range(1, base_period + 1)
            if base_period % d == 0 and t_min <= d <= t_max]
    return sorted(divs)


def _snap_log(value: float, sorted_set: List[int]) -> int:
    """Привязать value к ближайшему по логарифмической шкале элементу набора."""
    lv = math.log(value)
    return min(sorted_set, key=lambda x: abs(math.log(x) - lv))


def log_uniform_periods(n: int, t_min: float, t_max: float,
                        rng: np.random.Generator,
                        period_mode: str = "harmonic",
                        base_period: int = 2400) -> List[int]:
    """
    Генерация n периодов в диапазоне [t_min, t_max] (мс).

    Логравномерное распределение (Emberson, Stafford, Davis, 2010) даёт равное
    число задач в каждой декаде периодов, что лучше отражает реальные нагрузки,
    чем линейно-равномерное, смещённое в сторону больших периодов.

    period_mode:
      'log-uniform' -- непрерывные логравномерные периоды (макс. реализм, но
                       гиперпериод не ограничен; пригодно для аналитических
                       тестов выполнимости);
      'harmonic'    -- логравномерная выборка с привязкой к делителям
                       base_period (ограниченный гиперпериод; пригодно для
                       имитационного моделирования).
    """
    log_lo, log_hi = math.log(t_min), math.log(t_max)
    raw = np.exp(rng.uniform(log_lo, log_hi, size=n))
    if period_mode == "log-uniform":
        return [max(int(round(p)), int(t_min)) for p in raw]
    elif period_mode == "harmonic":
        pset = divisor_period_set(base_period, t_min, t_max)
        return [_snap_log(float(p), pset) for p in raw]
    else:
        raise ValueError(f"Неизвестный режим периодов: {period_mode}")


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def hyperperiod(periods: List[int]) -> int:
    """Гиперпериод набора -- НОК периодов задач."""
    h = 1
    for p in periods:
        h = _lcm(h, p)
    return h


# --------------------------------------------------------------------------- #
#  Структуры данных                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """Задача tau_i = (C_i, T_i, D_i, kind, phase).

    kind  -- 'P' (периодическая) или 'S' (спорадическая); для спорадической
             задачи T_i трактуется как минимальный межинтервал поступления;
    phase -- фаза phi_i (момент первого выпуска, мс). При phase = 0 для всех
             задач реализуется синхронный критический момент (худший случай)."""
    C: float          # WCET, мс
    T: int            # период / мин. межинтервал, мс
    D: int            # относительный дедлайн, мс
    kind: str = "P"   # 'P' | 'S'
    phase: int = 0    # фаза phi_i, мс

    def utilization(self) -> float:
        return self.C / self.T

    def density(self) -> float:
        return self.C / min(self.T, self.D)


@dataclass
class TaskSet:
    """Набор задач с метаданными порождения."""
    tasks: List[Task]
    m: int                    # число процессоров (ядер)
    u_norm: float             # нормированная загрузка U / m
    deadline_type: str        # 'implicit' | 'constrained'
    period_mode: str          # 'harmonic' | 'log-uniform'
    seed: int                 # seed данного набора
    set_id: int               # порядковый номер в точке сетки
    task_model: str = "periodic"   # 'periodic' | 'sporadic' | 'mixed'

    def total_u(self) -> float:
        return sum(t.utilization() for t in self.tasks)

    def hyperperiod(self) -> int:
        return hyperperiod([t.T for t in self.tasks])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "m": self.m,
            "u_norm": round(self.u_norm, 4),
            "deadline_type": self.deadline_type,
            "period_mode": self.period_mode,
            "task_model": self.task_model,
            "seed": self.seed,
            "set_id": self.set_id,
            "n": len(self.tasks),
            "total_u": round(self.total_u(), 6),
            "hyperperiod": self.hyperperiod(),
            # Формат tasks сохранён как тройки [C, T, D] для совместимости;
            # тип задачи и фаза вынесены в параллельные массивы.
            "tasks": [[round(t.C, 4), t.T, t.D] for t in self.tasks],
            "kinds": [t.kind for t in self.tasks],
            "phases": [t.phase for t in self.tasks],
        }


# --------------------------------------------------------------------------- #
#  Генерация одного набора                                                    #
# --------------------------------------------------------------------------- #
def generate_taskset(n: int, m: int, u_norm: float, deadline_type: str,
                     seed: int, set_id: int,
                     t_min: float, t_max: float,
                     period_mode: str = "harmonic",
                     base_period: int = 2400,
                     task_model: str = "periodic",
                     sporadic_fraction: float = 0.3,
                     offset_mode: str = "zero") -> TaskSet:
    """
    Сгенерировать один набор из n задач при заданной нормированной загрузке
    u_norm = U_total / m на m ядрах.

    task_model:
      'periodic' -- все задачи периодические (базовая модель);
      'sporadic' -- все задачи спорадические (T_i -- мин. межинтервал);
      'mixed'    -- доля sporadic_fraction задач спорадические, прочие
                    периодические.
    offset_mode:
      'zero'   -- фаза phi_i = 0 у всех задач (синхронный критический момент,
                  худший случай для анализа выполнимости);
      'random' -- phi_i ~ U(0, T_i) (выборочный сценарий; для спорадических
                  задач отражает произвольность моментов поступления).
    """
    rng = np.random.default_rng(seed)
    total_u = u_norm * m

    u = uunifast_discard(n, total_u, rng)
    periods = log_uniform_periods(n, t_min, t_max, rng,
                                  period_mode=period_mode,
                                  base_period=base_period)

    # Назначение типа задачи (детерминированно по индексу).
    if task_model == "periodic":
        kinds = ["P"] * n
    elif task_model == "sporadic":
        kinds = ["S"] * n
    elif task_model == "mixed":
        n_spor = int(round(sporadic_fraction * n))
        kinds = ["S"] * n_spor + ["P"] * (n - n_spor)
        rng.shuffle(kinds)
    else:
        raise ValueError(f"Неизвестная модель задач: {task_model}")

    tasks: List[Task] = []
    for ui, Ti, kind in zip(u, periods, kinds):
        Ci = ui * Ti
        if deadline_type == "implicit":
            Di = Ti
        elif deadline_type == "constrained":
            # C_i <= D_i <= T_i; нижняя граница -- ceil(C_i), чтобы дедлайн
            # был достижим хотя бы теоретически.
            lo = max(int(math.ceil(Ci)), 1)
            Di = int(rng.integers(lo, Ti + 1)) if lo < Ti else Ti
        else:
            raise ValueError(f"Неизвестный тип дедлайна: {deadline_type}")

        if offset_mode == "zero":
            phi = 0
        elif offset_mode == "random":
            phi = int(rng.integers(0, Ti))
        else:
            raise ValueError(f"Неизвестный режим фаз: {offset_mode}")

        tasks.append(Task(C=Ci, T=Ti, D=Di, kind=kind, phase=phi))

    return TaskSet(tasks=tasks, m=m, u_norm=u_norm,
                   deadline_type=deadline_type, period_mode=period_mode,
                   seed=seed, set_id=set_id, task_model=task_model)


# --------------------------------------------------------------------------- #
#  Генерация корпуса                                                          #
# --------------------------------------------------------------------------- #
def generate_corpus(out_dir: str,
                    m_values: List[int],
                    u_norm_values: List[float],
                    deadline_types: List[str],
                    n_sets: int,
                    tasks_per_core: int,
                    t_min: float,
                    t_max: float,
                    base_seed: int,
                    period_mode: str = "harmonic",
                    base_period: int = 2400,
                    task_model: str = "periodic",
                    sporadic_fraction: float = 0.3,
                    offset_mode: str = "zero") -> Dict[str, Any]:
    """
    Сгенерировать полный корпус по факторной сетке и сохранить по одному
    gzip-JSON-файлу на каждое значение m. Возвращает манифест.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "generator": "UUniFast-Discard + log-uniform periods",
        "references": ["Bini&Buttazzo2005", "Davis&Burns2009",
                       "Emberson2010"],
        "base_seed": base_seed,
        "m_values": m_values,
        "u_norm_values": [round(u, 4) for u in u_norm_values],
        "deadline_types": deadline_types,
        "n_sets_per_point": n_sets,
        "tasks_per_core": tasks_per_core,
        "period_range_ms": [t_min, t_max],
        "period_mode": period_mode,
        "base_period_ms": base_period if period_mode == "harmonic" else None,
        "task_model": task_model,
        "sporadic_fraction": sporadic_fraction if task_model == "mixed" else None,
        "offset_mode": offset_mode,
        "files": {},
        "totals": {},
    }

    total_sets = 0
    total_tasks = 0
    max_hp = 0

    for m in m_values:
        n = tasks_per_core * m
        m_records: List[Dict[str, Any]] = []
        for dl in deadline_types:
            for u_norm in u_norm_values:
                for k in range(n_sets):
                    # Детерминированный seed для каждой точки/набора.
                    seed = (base_seed
                            + m * 1_000_000
                            + deadline_types.index(dl) * 100_000
                            + int(round(u_norm * 100)) * 1_000
                            + k)
                    ts = generate_taskset(n, m, u_norm, dl, seed, k,
                                          t_min, t_max,
                                          period_mode=period_mode,
                                          base_period=base_period,
                                          task_model=task_model,
                                          sporadic_fraction=sporadic_fraction,
                                          offset_mode=offset_mode)
                    rec = ts.to_dict()
                    m_records.append(rec)
                    total_tasks += len(ts.tasks)
                    max_hp = max(max_hp, rec["hyperperiod"])
        total_sets += len(m_records)

        fname = f"tasksets_m{m}.json.gz"
        fpath = os.path.join(out_dir, fname)
        with gzip.open(fpath, "wt", encoding="utf-8") as f:
            json.dump({"m": m, "n_per_set": n, "records": m_records}, f,
                      ensure_ascii=False)
        manifest["files"][str(m)] = fname
        print(f"  m={m}: {len(m_records)} наборов, {n} задач/набор "
              f"-> {fname}")

    manifest["totals"] = {"task_sets": total_sets, "tasks": total_tasks,
                          "max_hyperperiod_ms": max_hp}

    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Небольшой читаемый образец для инспекции (m=2, U/m=0.70, implicit).
    sample = generate_taskset(2 * tasks_per_core, 2, 0.70, "implicit",
                              base_seed + 999, 0, t_min, t_max,
                              period_mode=period_mode, base_period=base_period,
                              task_model=task_model,
                              sporadic_fraction=sporadic_fraction,
                              offset_mode=offset_mode)
    with open(os.path.join(out_dir, "sample_m2_u070_implicit.json"), "w",
              encoding="utf-8") as f:
        json.dump(sample.to_dict(), f, ensure_ascii=False, indent=2)

    return manifest


def load_corpus_file(path: str) -> Dict[str, Any]:
    """Загрузить gzip-JSON-файл корпуса."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Генератор наборов задач РВ.")
    p.add_argument("--out", default="../tasksets", help="каталог вывода")
    p.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n-sets", type=int, default=200,
                   help="наборов на точку сетки (для эксперимента: 1000)")
    p.add_argument("--tasks-per-core", type=int, default=5)
    p.add_argument("--u-min", type=float, default=0.05)
    p.add_argument("--u-max", type=float, default=0.95)
    p.add_argument("--u-step", type=float, default=0.05)
    p.add_argument("--t-min", type=float, default=10.0)
    p.add_argument("--t-max", type=float, default=1000.0)
    p.add_argument("--deadlines", nargs="+",
                   default=["implicit", "constrained"])
    p.add_argument("--period-mode", default="harmonic",
                   choices=["harmonic", "log-uniform"])
    p.add_argument("--base-period", type=int, default=2400)
    p.add_argument("--task-model", default="periodic",
                   choices=["periodic", "sporadic", "mixed"],
                   help="модель нагрузки: периодическая, спорадическая "
                        "или смешанная")
    p.add_argument("--sporadic-fraction", type=float, default=0.3,
                   help="доля спорадических задач для --task-model mixed")
    p.add_argument("--offset-mode", default="zero",
                   choices=["zero", "random"],
                   help="фазы phi_i: zero -- синхронный критический момент, "
                        "random -- случайные фазы")
    p.add_argument("--seed", type=int, default=20260611)
    args = p.parse_args()

    u_values = [round(args.u_min + i * args.u_step, 4)
                for i in range(int(round((args.u_max - args.u_min)
                                          / args.u_step)) + 1)]

    print("Генерация корпуса наборов задач:")
    print(f"  m = {args.m}")
    print(f"  U/m = {u_values}")
    print(f"  дедлайны = {args.deadlines}")
    print(f"  {args.n_sets} наборов/точку, {args.tasks_per_core} задач/ядро")
    print(f"  периоды в [{args.t_min}, {args.t_max}] мс, "
          f"режим={args.period_mode}, база={args.base_period}, "
          f"seed={args.seed}")
    print(f"  модель задач={args.task_model}"
          + (f" (доля spor.={args.sporadic_fraction})"
             if args.task_model == "mixed" else "")
          + f", фазы={args.offset_mode}")

    manifest = generate_corpus(
        out_dir=args.out,
        m_values=args.m,
        u_norm_values=u_values,
        deadline_types=args.deadlines,
        n_sets=args.n_sets,
        tasks_per_core=args.tasks_per_core,
        t_min=args.t_min,
        t_max=args.t_max,
        base_seed=args.seed,
        period_mode=args.period_mode,
        base_period=args.base_period,
        task_model=args.task_model,
        sporadic_fraction=args.sporadic_fraction,
        offset_mode=args.offset_mode,
    )

    print(f"Итого: {manifest['totals']['task_sets']} наборов, "
          f"{manifest['totals']['tasks']} задач, "
          f"макс. гиперпериод {manifest['totals']['max_hyperperiod_ms']} мс.")


if __name__ == "__main__":
    main()
