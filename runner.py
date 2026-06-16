# -*- coding: utf-8 -*-
"""
runner.py -- прогон одного набора задач в SimSo и сбор метрик.

Запускает имитационную модель для пары (набор задач, планировщик) и
возвращает словарь измерений: пропуски дедлайнов, времена отклика, джиттер,
число вытеснений и миграций. Требует установленного SimSo.

Извлечение результатов опирается на объект model.results SimSo. Имена
атрибутов соответствуют документированному API SimSo (Chéramy et al., 2014);
доступ выполняется через getattr с запасными значениями, чтобы код был
устойчив к незначительным расхождениям версий.
"""

from __future__ import annotations

from typing import Dict, Any, List

from loader import build_configuration


def _safe(obj, attr, default=0):
    return getattr(obj, attr, default)


def run_one(record: Dict[str, Any], scheduler: str) -> Dict[str, Any]:
    """
    Прогнать один набор задач под заданным планировщиком.

    Возвращает словарь:
      schedulable     -- bool, ни один дедлайн не пропущен;
      deadline_misses -- int, число пропущенных дедлайнов;
      jobs_total      -- int, всего заданий;
      response_times  -- list[float], времена отклика всех заданий (мс);
      wcrt_per_task   -- list[float], худшее время отклика по каждой задаче;
      jitter_per_task -- list[float], джиттер отклика по каждой задаче;
      preemptions     -- int, число вытеснений;
      migrations      -- int, число миграций.
    """
    from simso.core import Model

    conf = build_configuration(record, scheduler)
    model = Model(conf)
    model.run_model()

    # Коэффициент перевода тактов в миллисекунды: SimSo считает время в тактах,
    # а времена отклика и джиттер в отчёте удобнее приводить в мс.
    cpms = float(getattr(model, "cycles_per_ms",
                         getattr(conf, "cycles_per_ms", 1.0))) or 1.0

    response_times: List[float] = []
    wcrt_per_task: List[float] = []
    jitter_per_task: List[float] = []
    deadline_misses = 0
    jobs_total = 0
    preemptions = 0
    migrations = 0

    # model.results.tasks: отображение Task -> TaskR (результаты по задаче).
    tasks_results = model.results.tasks
    iterable = (tasks_results.values()
                if hasattr(tasks_results, "values") else tasks_results)

    for task_r in iterable:
        rts: List[float] = []
        for job in _safe(task_r, "jobs", []):
            jobs_total += 1
            rt = _safe(job, "response_time", None)
            if rt is not None:
                rts.append(float(rt) / cpms)   # такты -> мс
            # Пропуск дедлайна: задание прервано либо превысило дедлайн.
            if _safe(job, "aborted", False) or _safe(job, "exceeded_deadline", 0):
                deadline_misses += 1
        if rts:
            response_times.extend(rts)
            wcrt_per_task.append(max(rts))
            jitter_per_task.append(max(rts) - min(rts))
        preemptions += int(_safe(task_r, "preemption_count", 0))
        migrations += int(_safe(task_r, "migration_count", 0))

    return {
        "scheduler": scheduler,
        "m": record["m"],
        "u_norm": record["u_norm"],
        "deadline_type": record["deadline_type"],
        "n": record["n"],
        "schedulable": deadline_misses == 0,
        "deadline_misses": deadline_misses,
        "jobs_total": jobs_total,
        "response_times": response_times,
        "wcrt_per_task": wcrt_per_task,
        "jitter_per_task": jitter_per_task,
        "preemptions": preemptions,
        "migrations": migrations,
    }
