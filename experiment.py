#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment.py -- драйвер серии экспериментов (этапы 4-5).

Прогоняет факторную сетку корпуса по набору алгоритмов планирования,
агрегирует метрики раздела 2.1 и сохраняет результаты в CSV, а также строит
кривые выполнимости S_A(U/m).

Два движка оценки выполнимости (--engine):

  analytic  -- аналитические тесты выполнимости (модуль feasibility.py).
               Работает БЕЗ установленного SimSo. Точные тесты для
               одноядерных EDF/RM/DM и для partitioned-EDF (по ядрам);
               для global/clustered-EDF используется достаточный тест GFB,
               поэтому их кривые выполнимости являются нижними оценками.

  simso     -- полное имитационное моделирование (модуль runner.py).
               Требует установленного SimSo (pip install simso). Даёт также
               времена отклика, джиттер, число вытеснений и миграций.

Примеры:
  # Быстрый аналитический прогон всей сетки (без SimSo):
  python experiment.py --engine analytic

  # Имитация для m=4, только глобальный и разделённый EDF, по 100 наборов:
  python experiment.py --engine simso --m 4 \\
        --algorithms G_EDF P_EDF_FF --max-sets 100 --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

from taskset_generator import load_corpus_file
import feasibility as fz
import metrics as mx


# --------------------------------------------------------------------------- #
#  Какие алгоритмы поддерживает каждый движок и для каких m                    #
# --------------------------------------------------------------------------- #
MONO = ["EDF_mono", "RM_mono", "DM_mono"]
MULTI = ["G_EDF", "G_RM", "P_EDF_FF", "P_EDF_BF", "P_EDF_WF", "P_EDF_NF",
         "C_EDF", "G_EDZL"]

# Алгоритмы, для которых реализован аналитический тест.
ANALYTIC_SUPPORTED = {"EDF_mono", "RM_mono", "DM_mono", "G_EDF",
                      "P_EDF_FF", "P_EDF_BF", "P_EDF_WF", "P_EDF_NF", "C_EDF"}


def default_algorithms(m: int, engine: str) -> List[str]:
    """Разумный набор алгоритмов по умолчанию для данного m и движка."""
    algos = MONO if m == 1 else [a for a in MULTI if a != "G_EDZL"]
    if engine == "analytic":
        algos = [a for a in algos if a in ANALYTIC_SUPPORTED]
    return algos


# --------------------------------------------------------------------------- #
#  Аналитический движок                                                        #
# --------------------------------------------------------------------------- #
def _pack(tasks: List[Tuple[float, int, int]], m: int, heuristic: str):
    """Упаковка задач по m ядрам (зеркалит эвристики partitioned.py).
    Возвращает список ядер (каждое -- список задач) либо None, если задача
    не помещается."""
    bins = [[] for _ in range(m)]          # задачи на каждом ядре
    load = [0.0] * m
    idx = list(range(len(tasks)))
    if heuristic in ("first-fit", "best-fit", "worst-fit"):
        idx.sort(key=lambda i: tasks[i][0] / tasks[i][1], reverse=True)
    nxt = 0
    for i in idx:
        u = tasks[i][0] / tasks[i][1]
        chosen = None
        if heuristic == "first-fit":
            for k in range(m):
                if load[k] + u <= 1.0 + 1e-9:
                    chosen = k
                    break
        elif heuristic == "best-fit":
            cand = [k for k in range(m) if load[k] + u <= 1.0 + 1e-9]
            chosen = min(cand, key=lambda k: 1.0 - load[k] - u) if cand else None
        elif heuristic == "worst-fit":
            cand = [k for k in range(m) if load[k] + u <= 1.0 + 1e-9]
            chosen = max(cand, key=lambda k: 1.0 - load[k] - u) if cand else None
        elif heuristic == "next-fit":
            for j in range(m):
                k = (nxt + j) % m
                if load[k] + u <= 1.0 + 1e-9:
                    chosen = k
                    nxt = k
                    break
        if chosen is None:
            return None
        bins[chosen].append(tasks[i])
        load[chosen] += u
    return bins


_HEUR = {"P_EDF_FF": "first-fit", "P_EDF_BF": "best-fit",
         "P_EDF_WF": "worst-fit", "P_EDF_NF": "next-fit"}


def analytic_schedulable(tasks: List[Tuple[float, int, int]], m: int,
                         algo: str, deadline_type: str) -> bool:
    """Аналитическая оценка выполнимости набора под алгоритмом algo.

    Тесты выполнимости (RTA, тест плотности/спроса EDF, граница Лю-Лейланда,
    GFB) опираются на худший случай -- синхронный критический момент. Поэтому
    при D <= T они дают одинаковый результат для периодических и спорадических
    задач (для спорадической задачи T -- минимальный межинтервал, и её худший
    сценарий совпадает с периодическим выпуском с максимальной частотой).
    Следовательно, аналитический движок применим к моделям periodic, sporadic
    и mixed без изменений, и тип задачи в нём не учитывается отдельно."""
    implicit = (deadline_type == "implicit")

    if algo == "EDF_mono":
        return (fz.edf_uniproc_implicit(tasks) if implicit
                else fz.edf_uniproc_exact(tasks))
    if algo == "RM_mono":
        return fz.fp_rta_test(tasks, "RM")
    if algo == "DM_mono":
        return fz.fp_rta_test(tasks, "DM")
    if algo == "G_EDF":
        # Достаточный тест GFB (нижняя оценка выполнимости).
        return fz.gedf_gfb_test(tasks, m)
    if algo in _HEUR:
        bins = _pack(tasks, m, _HEUR[algo])
        if bins is None:
            return False
        for b in bins:
            ok = (fz.edf_uniproc_implicit(b) if implicit
                  else fz.edf_uniproc_exact(b))
            if not ok:
                return False
        return True
    if algo == "C_EDF":
        # Кластеры по 2 ядра; распределение first-fit по загрузке;
        # внутри кластера -- достаточный тест GFB.
        cs = 2
        n_clusters = max(1, math.ceil(m / cs))
        sizes = [min(cs, m - c * cs) for c in range(n_clusters)]
        cload = [0.0] * n_clusters
        cbins: List[List] = [[] for _ in range(n_clusters)]
        for t in tasks:
            u = t[0] / t[1]
            placed = False
            for c in range(n_clusters):
                if cload[c] + u <= sizes[c] + 1e-9:
                    cbins[c].append(t)
                    cload[c] += u
                    placed = True
                    break
            if not placed:
                c = min(range(n_clusters), key=lambda x: cload[x])
                cbins[c].append(t)
                cload[c] += u
        for c in range(n_clusters):
            if cbins[c] and not fz.gedf_gfb_test(cbins[c], sizes[c]):
                return False
        return True
    raise ValueError(f"Аналитический тест для {algo} не реализован")


# --------------------------------------------------------------------------- #
#  Единый формат результата прогона                                           #
# --------------------------------------------------------------------------- #
def _empty_result(record, algo, schedulable, error=""):
    return {
        "scheduler": algo, "m": record["m"], "u_norm": record["u_norm"],
        "deadline_type": record["deadline_type"], "n": record["n"],
        "set_id": record["set_id"],
        "schedulable": bool(schedulable),
        "deadline_misses": 0 if schedulable else 1,
        "jobs_total": 1,
        "response_times": [], "wcrt_per_task": [], "jitter_per_task": [],
        "preemptions": 0, "migrations": 0, "error": error,
    }


def _worker(job: Tuple[Dict[str, Any], str, str]) -> Dict[str, Any]:
    """Один прогон (набор, алгоритм, движок). Верхнего уровня -- для
    распараллеливания ProcessPoolExecutor."""
    record, algo, engine = job
    tasks = [tuple(t) for t in record["tasks"]]
    try:
        if engine == "analytic":
            sched = analytic_schedulable(tasks, record["m"], algo,
                                         record["deadline_type"])
            return _empty_result(record, algo, sched)
        else:
            from runner import run_one
            res = run_one(record, algo)
            res["set_id"] = record["set_id"]
            res["error"] = ""
            return res
    except Exception as e:                       # noqa: BLE001
        # Для partitioned неудача упаковки = невыполнимость; прочие ошибки
        # фиксируются в поле error.
        r = _empty_result(record, algo, False, error=type(e).__name__ + ": "
                          + str(e))
        return r


# --------------------------------------------------------------------------- #
#  Сбор и агрегирование                                                       #
# --------------------------------------------------------------------------- #
class Aggregator:
    """Накопитель сводных метрик по ключу (m, deadline_type, algo, u_norm)."""

    def __init__(self):
        self.acc: Dict[Tuple, Dict[str, float]] = {}

    def add(self, res: Dict[str, Any]):
        key = (res["m"], res["deadline_type"], res["scheduler"], res["u_norm"])
        a = self.acc.setdefault(key, {
            "n": 0, "feasible": 0, "misses": 0, "jobs": 0,
            "pre": 0, "mig": 0, "wcrt_sum": 0.0, "wcrt_n": 0,
            "jit_sum": 0.0, "jit_n": 0, "errors": 0,
        })
        a["n"] += 1
        a["feasible"] += 1 if res["schedulable"] else 0
        a["misses"] += res.get("deadline_misses", 0)
        a["jobs"] += res.get("jobs_total", 0)
        a["pre"] += res.get("preemptions", 0)
        a["mig"] += res.get("migrations", 0)
        if res.get("error"):
            a["errors"] += 1
        for w in res.get("wcrt_per_task", []):
            a["wcrt_sum"] += w
            a["wcrt_n"] += 1
        for j in res.get("jitter_per_task", []):
            a["jit_sum"] += j
            a["jit_n"] += 1

    def aggregated_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for (m, dl, algo, u), a in sorted(self.acc.items()):
            rows.append({
                "m": m, "deadline_type": dl, "algorithm": algo, "u_norm": u,
                "n_sets": a["n"],
                "sched_ratio": a["feasible"] / a["n"] if a["n"] else float("nan"),
                "dmr": a["misses"] / a["jobs"] if a["jobs"] else float("nan"),
                "mean_preemptions": a["pre"] / a["n"] if a["n"] else float("nan"),
                "mean_migrations": a["mig"] / a["n"] if a["n"] else float("nan"),
                "mean_wcrt": a["wcrt_sum"] / a["wcrt_n"] if a["wcrt_n"]
                             else float("nan"),
                "mean_jitter": a["jit_sum"] / a["jit_n"] if a["jit_n"]
                               else float("nan"),
                "errors": a["errors"],
            })
        return rows

    def summary_rows(self) -> List[Dict[str, Any]]:
        # Группируем по (m, dl, algo), собираем серию по u.
        series: Dict[Tuple, List[Tuple[float, float]]] = {}
        for (m, dl, algo, u), a in self.acc.items():
            ratio = a["feasible"] / a["n"] if a["n"] else float("nan")
            series.setdefault((m, dl, algo), []).append((u, ratio))
        rows = []
        for (m, dl, algo), pts in sorted(series.items()):
            pts.sort()
            us = [p[0] for p in pts]
            ss = [p[1] for p in pts]
            rows.append({
                "m": m, "deadline_type": dl, "algorithm": algo,
                "breakdown_utilization": mx.breakdown_utilization(us, ss),
                "weighted_schedulability": mx.weighted_schedulability(us, ss),
            })
        return rows


# --------------------------------------------------------------------------- #
#  Построение графиков                                                        #
# --------------------------------------------------------------------------- #
def plot_curves(agg_rows: List[Dict[str, Any]], out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # pragma: no cover
        print(f"matplotlib недоступен ({e}); графики пропущены.")
        return []

    os.makedirs(out_dir, exist_ok=True)
    # Группировка по (m, deadline_type).
    groups: Dict[Tuple, Dict[str, List[Tuple[float, float]]]] = {}
    for r in agg_rows:
        g = groups.setdefault((r["m"], r["deadline_type"]), {})
        g.setdefault(r["algorithm"], []).append((r["u_norm"], r["sched_ratio"]))

    files = []
    for (m, dl), algos in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for algo, pts in sorted(algos.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=algo)
        ax.set_xlabel("Нормированная загрузка U/m")
        ax.set_ylabel("Коэффициент выполнимости S_A")
        ax.set_title(f"Кривые выполнимости: m={m}, дедлайны: {dl}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fpath = os.path.join(out_dir, f"sched_m{m}_{dl}.png")
        fig.savefig(fpath, dpi=130)
        plt.close(fig)
        files.append(fpath)
    return files


# --------------------------------------------------------------------------- #
#  Запись CSV                                                                  #
# --------------------------------------------------------------------------- #
def write_csv(path: str, rows: List[Dict[str, Any]], fields: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
#  Основной сценарий                                                          #
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description="Драйвер серии экспериментов.")
    p.add_argument("--engine", choices=["analytic", "simso"],
                   default="analytic")
    p.add_argument("--corpus-dir", default="../tasksets")
    p.add_argument("--out-dir", default="../results")
    p.add_argument("--m", type=int, nargs="+", default=None,
                   help="какие m обрабатывать (по умолчанию все из манифеста)")
    p.add_argument("--deadlines", nargs="+", default=None,
                   help="implicit / constrained (по умолчанию оба)")
    p.add_argument("--algorithms", nargs="+", default=None,
                   help="список алгоритмов (по умолчанию -- авто по m)")
    p.add_argument("--max-sets", type=int, default=None,
                   help="ограничить число наборов на точку сетки")
    p.add_argument("--jobs", type=int, default=1,
                   help="число параллельных процессов")
    p.add_argument("--save-runs", action="store_true",
                   help="сохранить построчные результаты прогонов")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.corpus_dir, "manifest.json")
    import json
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    m_values = args.m or manifest["m_values"]
    deadlines = args.deadlines or manifest["deadline_types"]

    if args.engine == "simso":
        try:
            import simso  # noqa: F401
        except ImportError:
            print("SimSo не установлен. Установите (pip install simso) или "
                  "используйте --engine analytic.")
            return 1

    # Сбор заданий.
    jobs: List[Tuple[Dict[str, Any], str, str]] = []
    for m in m_values:
        algos = args.algorithms or default_algorithms(m, args.engine)
        if args.engine == "analytic":
            algos = [a for a in algos if a in ANALYTIC_SUPPORTED]
        corpus = load_corpus_file(os.path.join(
            args.corpus_dir, manifest["files"][str(m)]))
        # Группируем по (deadline_type, u_norm) для ограничения max-sets.
        from collections import defaultdict
        buckets = defaultdict(list)
        for rec in corpus["records"]:
            if rec["deadline_type"] in deadlines:
                buckets[(rec["deadline_type"], rec["u_norm"])].append(rec)
        for key, recs in buckets.items():
            chosen = recs[:args.max_sets] if args.max_sets else recs
            for rec in chosen:
                for algo in algos:
                    jobs.append((rec, algo, args.engine))

    total = len(jobs)
    print(f"Движок: {args.engine}; заданий к прогону: {total}")
    if total == 0:
        print("Нет заданий (проверьте параметры).")
        return 1

    agg = Aggregator()
    run_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    done = 0

    def handle(res):
        nonlocal done
        agg.add(res)
        if args.save_runs:
            run_rows.append({
                "m": res["m"], "deadline_type": res["deadline_type"],
                "u_norm": res["u_norm"], "set_id": res.get("set_id", -1),
                "algorithm": res["scheduler"],
                "schedulable": int(res["schedulable"]),
                "deadline_misses": res.get("deadline_misses", 0),
                "jobs_total": res.get("jobs_total", 0),
                "preemptions": res.get("preemptions", 0),
                "migrations": res.get("migrations", 0),
                "error": res.get("error", ""),
            })
        done += 1
        if done % max(1, total // 20) == 0 or done == total:
            el = time.time() - t0
            print(f"  {done}/{total} ({100*done/total:.0f}%), {el:.1f} с")

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for res in ex.map(_worker, jobs, chunksize=64):
                handle(res)
    else:
        for job in jobs:
            handle(_worker(job))

    # Запись результатов.
    agg_rows = agg.aggregated_rows()
    sum_rows = agg.summary_rows()

    agg_path = os.path.join(args.out_dir, "results_aggregated.csv")
    write_csv(agg_path, agg_rows, ["m", "deadline_type", "algorithm",
              "u_norm", "n_sets", "sched_ratio", "dmr", "mean_preemptions",
              "mean_migrations", "mean_wcrt", "mean_jitter", "errors"])
    sum_path = os.path.join(args.out_dir, "results_summary.csv")
    write_csv(sum_path, sum_rows, ["m", "deadline_type", "algorithm",
              "breakdown_utilization", "weighted_schedulability"])
    print(f"Сводка по точкам: {agg_path}")
    print(f"Итоговые показатели: {sum_path}")
    if args.save_runs:
        runs_path = os.path.join(args.out_dir, "results_runs.csv")
        write_csv(runs_path, run_rows, ["m", "deadline_type", "u_norm",
                  "set_id", "algorithm", "schedulable", "deadline_misses",
                  "jobs_total", "preemptions", "migrations", "error"])
        print(f"Построчные результаты: {runs_path}")

    if not args.no_plot:
        figs = plot_curves(agg_rows, os.path.join(args.out_dir, "figures"))
        for fp in figs:
            print(f"График: {fp}")

    # Короткая сводка в консоль.
    print("\nИтоговая взвешенная выполнимость W (по убыванию):")
    sum_rows.sort(key=lambda r: (r["m"], r["deadline_type"],
                                 -(r["weighted_schedulability"]
                                   if not math.isnan(r["weighted_schedulability"])
                                   else -1)))
    for r in sum_rows:
        print(f"  m={r['m']} {r['deadline_type']:<11} {r['algorithm']:<10} "
              f"W={r['weighted_schedulability']:.3f}  "
              f"U*={r['breakdown_utilization']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
