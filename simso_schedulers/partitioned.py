# -*- coding: utf-8 -*-
"""
partitioned.py -- partitioned-планирование с EDF на каждом ядре и четырьмя
эвристиками упаковки: First/Best/Worst/Next Fit (упрочнённое).

Реализация выполнена напрямую (без базового класса SimSo PartitionedScheduler),
что даёт полный контроль над обработкой прерываний (on_abort) на каждом ядре.
Каждая задача статически закрепляется ровно за одним ядром (миграций нет);
распределение -- эвристика упаковки в контейнеры (Bin Packing, NP-трудная).
На каждом ядре независимо работает EDF. Если задача не помещается ни на одно
ядро в пределах ёмкости (sum U_i <= 1), она назначается наименее загруженному
ядру -- тогда перегрузка проявится как пропуски дедлайнов при моделировании
(набор корректно классифицируется как невыполнимый).

Замена per-core EDF на RM/DM (вариант P-RM/P-DM) сводится к смене ключа _key.
"""

from simso.core import Scheduler
from simso.schedulers import scheduler


def _active(job):
    f = getattr(job, "is_active", None)
    return f() if callable(f) else True


def _util(task):
    return float(task.wcet) / float(task.period)


class _PartitionedBase(Scheduler):
    """База: статическое закрепление задач за ядрами + EDF на каждом ядре.
    HEURISTIC задаёт эвристику упаковки; _key -- приоритет (по умолчанию EDF)."""

    HEURISTIC = "first-fit"

    def _key(self, job):
        return job.absolute_deadline

    def init(self):
        procs = list(self.processors)
        m = len(procs)
        self.proc_index = {p.identifier: i for i, p in enumerate(procs)}
        self.ready = [[] for _ in range(m)]      # очередь готовности на ядро
        load = [0.0] * m

        tasks = list(self.task_list)
        if self.HEURISTIC in ("first-fit", "best-fit", "worst-fit"):
            tasks.sort(key=_util, reverse=True)

        self.assign = {}                          # task.identifier -> индекс ядра
        nxt = 0
        for task in tasks:
            u = _util(task)
            chosen = None
            if self.HEURISTIC == "first-fit":
                for k in range(m):
                    if load[k] + u <= 1.0 + 1e-9:
                        chosen = k
                        break
            elif self.HEURISTIC == "best-fit":
                cand = [k for k in range(m) if load[k] + u <= 1.0 + 1e-9]
                if cand:
                    chosen = min(cand, key=lambda k: 1.0 - load[k] - u)
            elif self.HEURISTIC == "worst-fit":
                cand = [k for k in range(m) if load[k] + u <= 1.0 + 1e-9]
                if cand:
                    chosen = max(cand, key=lambda k: 1.0 - load[k] - u)
            elif self.HEURISTIC == "next-fit":
                for j in range(m):
                    k = (nxt + j) % m
                    if load[k] + u <= 1.0 + 1e-9:
                        chosen = k
                        nxt = k
                        break
            else:
                raise ValueError(f"Неизвестная эвристика: {self.HEURISTIC}")

            if chosen is None:
                # Не помещается -> наименее загруженное ядро (перегрузка
                # проявится как пропуски дедлайнов при моделировании).
                chosen = min(range(m), key=lambda k: load[k])

            self.assign[task.identifier] = chosen
            load[chosen] += u

    def _core_of(self, job):
        return self.assign[job.task.identifier]

    def on_activate(self, job):
        c = self._core_of(job)
        self.ready[c].append(job)
        self.processors[c].resched()

    def _remove(self, job):
        c = self._core_of(job)
        if job in self.ready[c]:
            self.ready[c].remove(job)
        self.processors[c].resched()

    def on_terminated(self, job):
        self._remove(job)

    def on_abort(self, job):
        self._remove(job)

    def schedule(self, cpu):
        c = self.proc_index[cpu.identifier]
        ready = [j for j in self.ready[c] if _active(j)]
        job = min(ready, key=self._key) if ready else None
        return (job, cpu)


@scheduler("simso_schedulers.P_EDF_FF")
class P_EDF_FF(_PartitionedBase):
    """Partitioned EDF, First Fit (Decreasing)."""
    HEURISTIC = "first-fit"


@scheduler("simso_schedulers.P_EDF_BF")
class P_EDF_BF(_PartitionedBase):
    """Partitioned EDF, Best Fit (Decreasing)."""
    HEURISTIC = "best-fit"


@scheduler("simso_schedulers.P_EDF_WF")
class P_EDF_WF(_PartitionedBase):
    """Partitioned EDF, Worst Fit (Decreasing)."""
    HEURISTIC = "worst-fit"


@scheduler("simso_schedulers.P_EDF_NF")
class P_EDF_NF(_PartitionedBase):
    """Partitioned EDF, Next Fit."""
    HEURISTIC = "next-fit"
