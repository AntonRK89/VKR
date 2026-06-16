# -*- coding: utf-8 -*-
"""
clustered.py -- clustered EDF (C-EDF), упрочнённый.

Ядра объединяются в кластеры размера cluster_size; внутри кластера действует
глобальный EDF, между кластерами задачи распределяются статически (first-fit
по загрузке, ёмкость кластера = числу ядер). cluster_size = 1 эквивалентно
partitioned-EDF, cluster_size = m -- глобальному EDF.

Упрочнение: обработчик on_abort и фильтр активных заданий в schedule().
"""

from simso.core import Scheduler
from simso.schedulers import scheduler


def _active(job):
    f = getattr(job, "is_active", None)
    return f() if callable(f) else True


@scheduler("simso_schedulers.C_EDF")
class C_EDF(Scheduler):
    """Clustered EDF с настраиваемым размером кластера."""

    cluster_size = 2

    def init(self):
        cs = max(1, int(getattr(self, "cluster_size", 2)))
        procs = list(self.processors)

        self.clusters = []
        for i in range(0, len(procs), cs):
            grp = procs[i:i + cs]
            self.clusters.append({"procs": grp, "ready": [],
                                  "cap": float(len(grp)), "load": 0.0})

        self.proc_cluster = {}
        for cid, cl in enumerate(self.clusters):
            for p in cl["procs"]:
                self.proc_cluster[p.identifier] = cid

        # Статическое распределение задач по кластерам (First Fit по U).
        self.task_cluster = {}
        for task in self.task_list:
            u = float(task.wcet) / float(task.period)
            placed = False
            for cid, cl in enumerate(self.clusters):
                if cl["load"] + u <= cl["cap"] + 1e-9:
                    cl["load"] += u
                    self.task_cluster[task.identifier] = cid
                    placed = True
                    break
            if not placed:
                cid = min(range(len(self.clusters)),
                          key=lambda c: self.clusters[c]["load"])
                self.clusters[cid]["load"] += u
                self.task_cluster[task.identifier] = cid

    def _cluster_of(self, job):
        return self.task_cluster[job.task.identifier]

    def on_activate(self, job):
        cid = self._cluster_of(job)
        self.clusters[cid]["ready"].append(job)
        self.clusters[cid]["procs"][0].resched()

    def _remove(self, job):
        cid = self._cluster_of(job)
        ready = self.clusters[cid]["ready"]
        if job in ready:
            ready.remove(job)
        self.clusters[cid]["procs"][0].resched()

    def on_terminated(self, job):
        self._remove(job)

    def on_abort(self, job):
        self._remove(job)

    def schedule(self, cpu):
        cid = self.proc_cluster[cpu.identifier]
        cl = self.clusters[cid]
        procs = cl["procs"]
        ready = [j for j in cl["ready"] if _active(j)]
        decision = None
        if ready:
            def cpu_key(p):
                if not p.running:
                    return (1, 0)
                return (0, p.running.absolute_deadline)
            cpu_min = max(procs, key=cpu_key)
            job = min(ready, key=lambda x: x.absolute_deadline)
            if (cpu_min.running is None or
                    cpu_min.running.absolute_deadline > job.absolute_deadline):
                cl["ready"].remove(job)
                if cpu_min.running:
                    cl["ready"].append(cpu_min.running)
                decision = (job, cpu_min)
        return decision
