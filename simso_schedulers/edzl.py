# -*- coding: utf-8 -*-
"""
edzl.py -- Global Earliest Deadline until Zero Laxity (EDZL), опционально.

Гибрид динамических приоритетов: пока запас времени (laxity) положителен,
задание планируется по EDF; при нулевом запасе получает наивысший приоритет.
Запас: L_i(t) = d_i - t - e_i(t).

ВНИМАНИЕ. Вычисление запаса опирается на атрибуты SimSo (absolute_deadline,
ret) и текущее время; реализация экспериментальная и в основной серии не
используется. Упрочнение: on_abort и фильтр активных заданий в schedule().
"""

from simso.core import Scheduler
from simso.schedulers import scheduler


def _active(job):
    f = getattr(job, "is_active", None)
    return f() if callable(f) else True


@scheduler("simso_schedulers.G_EDZL")
class G_EDZL(Scheduler):
    """Global EDZL: EDF с абсолютным приоритетом для задач нулевого запаса."""

    def init(self):
        self.ready_list = []

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

    def _now_ms(self):
        return self.sim.now() / self.sim.cycles_per_ms

    def _prio_key(self, job):
        """Меньший ключ -> выше приоритет. Первый компонент: 0 для задач с
        нулевым/отрицательным запасом, 1 для остальных; второй -- дедлайн."""
        try:
            laxity = job.absolute_deadline - self._now_ms() - job.ret
        except AttributeError:
            laxity = 1.0
        zero = 0 if laxity <= 0 else 1
        return (zero, job.absolute_deadline)

    def schedule(self, cpu):
        decision = None
        ready = [j for j in self.ready_list if _active(j)]
        if ready:
            def cpu_key(p):
                if not p.running:
                    return (1, (1, 0))
                return (0, self._prio_key(p.running))
            cpu_min = max(self.processors, key=cpu_key)

            job = min(ready, key=self._prio_key)

            preempt = (cpu_min.running is None or
                       self._prio_key(job) < self._prio_key(cpu_min.running))
            if preempt:
                self.ready_list.remove(job)
                if cpu_min.running:
                    self.ready_list.append(cpu_min.running)
                decision = (job, cpu_min)
        return decision
