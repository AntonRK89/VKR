# -*- coding: utf-8 -*-
"""
global_sched.py -- глобальные планировщики G-EDF и G-RM (упрочнённые).

Единая очередь готовности на m ядер; на каждом вызове schedule(cpu)
принимается одно решение о вытеснении, повторные вызовы перепланирования
доводят систему до исполнения m старших по приоритету заданий.

Упрочнение: обработчик on_abort (удаление прерванного по пропуску дедлайна
задания из очереди) и выбор только активных заданий в schedule().
"""

from simso.core import Scheduler
from simso.schedulers import scheduler


def _active(job):
    f = getattr(job, "is_active", None)
    return f() if callable(f) else True


class _GlobalBase(Scheduler):
    """Общая логика глобального планирования. Подкласс задаёт _prio(job)
    (меньше -- приоритетнее) и _run_prio(running_job)."""

    def init(self):
        self.ready_list = []

    def _prio(self, job):
        raise NotImplementedError

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        if job in self.ready_list:
            self.ready_list.remove(job)
        job.cpu.resched()

    def on_abort(self, job):
        if job in self.ready_list:
            self.ready_list.remove(job)
        job.cpu.resched()

    def schedule(self, cpu):
        decision = None
        ready = [j for j in self.ready_list if _active(j)]
        if ready:
            # Ядро для вытеснения: свободное, иначе исполняющее задание с
            # наименее приоритетным (наибольший ключ) заданием.
            def cpu_key(p):
                if not p.running:
                    return (1, 0)
                return (0, self._prio(p.running))
            cpu_min = max(self.processors, key=cpu_key)

            job = min(ready, key=self._prio)

            if (cpu_min.running is None or
                    self._prio(cpu_min.running) > self._prio(job)):
                self.ready_list.remove(job)
                if cpu_min.running:
                    self.ready_list.append(cpu_min.running)
                decision = (job, cpu_min)
        return decision


@scheduler("simso_schedulers.G_EDF")
class G_EDF(_GlobalBase):
    """Global EDF: динамический приоритет по абсолютному дедлайну."""
    def _prio(self, job):
        return job.absolute_deadline


@scheduler("simso_schedulers.G_RM")
class G_RM(_GlobalBase):
    """Global Rate Monotonic: фиксированный приоритет по периоду."""
    def _prio(self, job):
        return job.period
