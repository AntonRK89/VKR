# -*- coding: utf-8 -*-
"""
Пакет реализаций планировщиков для имитационного моделирования в SimSo.

Каждый модуль содержит один или несколько классов-планировщиков,
наследующих simso.core.Scheduler (или его подкласс PartitionedScheduler) и
зарегистрированных декоратором @scheduler. Классы написаны под API SimSo
(Chéramy, Hladik, Déplanche, 2014) и предназначены для загрузки симулятором
по имени файла и имени класса (Configuration.scheduler_info).

Состав:
  mono.py        -- EDF_mono, RM_mono, DM_mono     (одно ядро)
  global_sched.py-- G_EDF, G_RM                    (глобальное планирование)
  partitioned.py -- P_EDF_FF/BF/WF/NF              (partitioned, 4 эвристики)
  clustered.py   -- C_EDF                          (clustered)
  edzl.py        -- G_EDZL                          (гибрид EDF/ZL, опционально)
"""
