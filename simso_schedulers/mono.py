# -*- coding: utf-8 -*-
"""
mono.py -- одноядерные планировщики: EDF, RM, DM (упрочнённые).

Помимо штатных обработчиков SimSo (on_activate / on_terminated / schedule)
реализован on_abort: при пропуске дедлайна SimSo прерывает (aborts) задание,
и его необходимо удалить из очереди готовности, иначе планировщик попытается
запустить уже терминированное задание ("Can't schedule a terminated job!").
Дополнительно schedule() выбирает только активные задания (is_active) -- это
страховка на случай, если прерванное задание по какой-либо причине осталось
в очереди.
"""

from simso.core import Scheduler
from simso.schedulers import scheduler


def _active(job):
    """Безопасная проверка активности задания (устойчива к версии SimSo)."""
    f = getattr(job, "is_active", None)
    return f() if callable(f) else True


class _MonoBase(Scheduler):
    """Общая логика одноядерных планировщиков. Подкласс задаёт _key(job)."""

    def init(self):
        self.ready_list = []

    def _key(self, job):
        raise NotImplementedError

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        if job in self.ready_list:
            self.ready_list.remove(job)
        job.cpu.resched()

    def on_abort(self, job):
        # Прерывание при пропуске дедлайна -- убрать задание из очереди.
        if job in self.ready_list:
            self.ready_list.remove(job)
        job.cpu.resched()

    def schedule(self, cpu):
        ready = [j for j in self.ready_list if _active(j)]
        job = min(ready, key=self._key) if ready else None
        return (job, cpu)


@scheduler("simso_schedulers.EDF_mono")
class EDF_mono(_MonoBase):
    """Earliest Deadline First: приоритет по наименьшему абсолютному дедлайну.
    Оптимален на одном ядре, выполним при U <= 1 (Liu, Layland, 1973)."""
    def _key(self, job):
        return job.absolute_deadline


@scheduler("simso_schedulers.RM_mono")
class RM_mono(_MonoBase):
    """Rate Monotonic: фиксированный приоритет по наименьшему периоду."""
    def _key(self, job):
        return job.period


@scheduler("simso_schedulers.DM_mono")
class DM_mono(_MonoBase):
    """Deadline Monotonic: фиксированный приоритет по наименьшему
    относительному дедлайну (Leung, Whitehead, 1982)."""
    def _key(self, job):
        return job.task.deadline
