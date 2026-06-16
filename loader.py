# -*- coding: utf-8 -*-
"""
loader.py -- построение конфигурации SimSo из записи корпуса.

Преобразует набор задач (формат генератора: словарь с полями m, tasks,
hyperperiod, ...) в объект simso.configuration.Configuration, готовый к
запуску. Требует установленного SimSo (pip install simso); при отсутствии
SimSo модуль импортируется, но build_configuration() выбросит ImportError.

Соответствие имён планировщиков и файлов задаётся словарём SCHEDULERS:
ключ -- короткое имя, значение -- (относительный путь к .py, имя класса).
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, Any, Tuple

# Каталог с реализациями планировщиков и его родитель (для импорта пакета).
_SCHED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "simso_schedulers")
_CODE_DIR = os.path.dirname(_SCHED_DIR)

# Реестр планировщиков: имя -> (модуль пакета simso_schedulers, имя класса).
SCHEDULERS: Dict[str, Tuple[str, str]] = {
    "EDF_mono": ("mono", "EDF_mono"),
    "RM_mono":  ("mono", "RM_mono"),
    "DM_mono":  ("mono", "DM_mono"),
    "G_EDF":    ("global_sched", "G_EDF"),
    "G_RM":     ("global_sched", "G_RM"),
    "P_EDF_FF": ("partitioned", "P_EDF_FF"),
    "P_EDF_BF": ("partitioned", "P_EDF_BF"),
    "P_EDF_WF": ("partitioned", "P_EDF_WF"),
    "P_EDF_NF": ("partitioned", "P_EDF_NF"),
    "C_EDF":    ("clustered", "C_EDF"),
    "G_EDZL":   ("edzl", "G_EDZL"),
}

# Кэш загруженных классов планировщиков (импорт модуля -- один раз на процесс).
_CLASS_CACHE: Dict[str, Any] = {}


def load_scheduler_class(name: str):
    """
    Импортировать и вернуть КЛАСС планировщика по имени.

    Возврат самого объекта класса (а не пути к файлу) обходит механизм SimSo,
    который загружает планировщик из файла и ищет вновь зарегистрированные
    классы; при многократной загрузке одного файла (несколько классов в файле,
    параллельные прогоны) этот механизм давал сбой. Передача готового класса
    в Configuration.scheduler_info.clas детерминирована и потоко-/процессо-
    безопасна.
    """
    if name in _CLASS_CACHE:
        return _CLASS_CACHE[name]
    if name not in SCHEDULERS:
        raise KeyError(f"Неизвестный планировщик: {name}")
    module_stem, class_name = SCHEDULERS[name]
    if _CODE_DIR not in sys.path:
        sys.path.insert(0, _CODE_DIR)
    module = importlib.import_module(f"simso_schedulers.{module_stem}")
    cls = getattr(module, class_name)
    _CLASS_CACHE[name] = cls
    return cls


def _sporadic_activations(T, phase, duration, jitter, rng):
    """
    Сформировать список моментов поступления спорадической задачи на интервале
    [0, duration] (мс).

    Межинтервал >= T (минимальный межинтервал). jitter -- относительный
    разброс задержки сверх минимума: gap = T + U(0, jitter*T). При jitter = 0
    задача поступает с максимальной частотой (каждые T) -- консервативный
    режим, эквивалентный периодическому с фазой phase; именно он соответствует
    худшему случаю для анализа выполнимости.
    """
    dates = []
    t = float(phase)
    while t < duration:
        dates.append(round(t, 3))
        extra = (rng.uniform(0.0, jitter * T) if jitter > 0 else 0.0)
        t += T + extra
    if not dates:                      # хотя бы одно поступление
        dates = [float(phase)]
    return dates


def build_configuration(record, scheduler, sporadic_jitter=0.0, seed=0):
    """
    Собрать Configuration SimSo для одного набора задач и планировщика.

    record          -- запись корпуса (см. taskset_generator.TaskSet.to_dict);
    scheduler       -- ключ из SCHEDULERS;
    sporadic_jitter -- относительный разброс межинтервалов спорадических задач
                       (0 -- поступление с максимальной частотой, худший
                       случай);
    seed            -- seed для воспроизводимого порождения моментов
                       поступления спорадических задач.
    """
    # Импорт SimSo отложен, чтобы модуль читался и без установленного SimSo.
    from simso.configuration import Configuration
    import numpy as np

    sched_cls = load_scheduler_class(scheduler)   # бросит KeyError, если нет

    m = record["m"]
    tasks = record["tasks"]                # [[C, T, D], ...]
    hyperperiod = record["hyperperiod"]    # мс
    # Тип задачи и фаза (с запасными значениями для старого корпуса).
    n = len(tasks)
    kinds = record.get("kinds", ["P"] * n)
    phases = record.get("phases", [0] * n)

    conf = Configuration()
    # Длительность прогона = один гиперпериод (в тактах).
    duration_ms = hyperperiod
    conf.duration = int(duration_ms * conf.cycles_per_ms)

    rng = np.random.default_rng(seed + record.get("set_id", 0))

    for i, ((C, T, D), kind, phase) in enumerate(
            zip(tasks, kinds, phases), start=1):
        if kind == "S":
            # Спорадическая задача: задаётся явным списком моментов выпуска.
            # period = T (минимальный межинтервал) указывается явно, чтобы
            # коэффициент использования wcet/period был корректен для
            # статического распределения в partitioned/clustered-планировании.
            dates = _sporadic_activations(T, phase, duration_ms,
                                          sporadic_jitter, rng)
            conf.add_task(
                name=f"T{i}",
                identifier=i,
                task_type="Sporadic",
                period=T,
                deadline=D,
                wcet=C,
                list_activation_dates=dates,
            )
        else:
            # Периодическая задача (фаза -- момент первого выпуска).
            conf.add_task(
                name=f"T{i}",
                identifier=i,
                task_type="Periodic",
                period=T,
                activation_date=phase,
                deadline=D,
                wcet=C,
            )

    for p in range(1, m + 1):
        conf.add_processor(name=f"CPU{p}", identifier=p)

    # Передаём готовый КЛАСС планировщика (а не путь к файлу): SimSo при
    # clas-как-классе использует его напрямую, минуя загрузку из файла.
    conf.scheduler_info.clas = sched_cls

    conf.check_all()   # валидация параметров набора и конфигурации
    return conf
