#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_smoke_test.py -- минимальная проверка работоспособности связки
loader+runner+планировщики при наличии установленного SimSo.

Берёт один небольшой набор задач из корпуса (m=2) и прогоняет его под
несколькими планировщиками, печатая сводку. Если SimSo не установлен,
выводит инструкцию по установке и корректно завершается.

Запуск:
    python run_smoke_test.py
"""

import os
import sys

from taskset_generator import load_corpus_file


def main():
    try:
        import simso  # noqa: F401
    except ImportError:
        print("SimSo не установлен. Установите его командой:")
        print("    pip install simso")
        print("После установки повторите запуск для имитационного "
              "моделирования.")
        return 0

    from runner import run_one

    here = os.path.dirname(os.path.abspath(__file__))
    corpus = load_corpus_file(os.path.join(here, "..", "tasksets",
                                           "tasksets_m2.json.gz"))
    # Берём первый набор средней загрузки.
    record = next(r for r in corpus["records"]
                  if abs(r["u_norm"] - 0.6) < 1e-6
                  and r["deadline_type"] == "implicit")

    print(f"Набор: m={record['m']}, n={record['n']}, "
          f"U/m={record['u_norm']}, гиперпериод={record['hyperperiod']} мс")
    print(f"{'планировщик':<10} {'выполним':<9} {'пропуски':<9} "
          f"{'вытесн.':<8} {'миграции':<8}")
    for sched in ["G_EDF", "P_EDF_FF", "P_EDF_BF", "C_EDF", "G_RM"]:
        try:
            res = run_one(record, sched)
            print(f"{sched:<10} {str(res['schedulable']):<9} "
                  f"{res['deadline_misses']:<9} {res['preemptions']:<8} "
                  f"{res['migrations']:<8}")
        except Exception as e:                       # pragma: no cover
            print(f"{sched:<10} ОШИБКА: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
