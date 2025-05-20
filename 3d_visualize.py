# -*- coding: utf-8 -*-
"""
Скрипт для создания 3D-визуализаций окружения лигандов
с использованием ИМПОРТИРУЕМОГО модуля PyMOL (API).
Требует установки PyMOL через Conda (pymol-bundle).
"""

import os
import pandas as pd
from tqdm import tqdm
import sys
from collections import defaultdict
import random
import time

# --- Попытка импорта PyMOL ---
try:
    # Инициализация PyMOL в режиме без GUI
    # Важно сделать до первого вызова cmd
    import pymol

    pymol.pymol_argv = ["pymol", "-qc"]  # Запуск в тихом режиме без GUI
    pymol.finish_launching()
    from pymol import cmd

    print("Модуль PyMOL успешно импортирован.")
    PYMOL_AVAILABLE = True
except ImportError:
    print("! Ошибка: Не удалось импортировать модуль 'pymol'.")
    print("  Убедитесь, что PyMOL установлен через Conda (pymol-bundle)")
    print(
        "  и интерпретатор Python для этого проекта настроен на соответствующее Conda-окружение."
    )
    PYMOL_AVAILABLE = False

    # Определим заглушку для cmd, чтобы скрипт не падал дальше
    class CmdMock:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                print(f"PyMOL не доступен. Вызов {name} проигнорирован.")
                # Для count_atoms вернем 0, чтобы if не падал
                if name == "count_atoms":
                    return 0
                # Для save/load/quit ничего не делаем
                pass

            return method

    cmd = CmdMock()


# --- Конфигурация ---
BASE_RESULTS_DIR = "analysis_results"
RING_DIR = os.path.join(BASE_RESULTS_DIR, "ring_subst_env")
FULL_CHAIN_DIR = os.path.join(BASE_RESULTS_DIR, "full_chain_env")
RAW_DATA_NAME = "neighbor_residues_raw.csv"
CIF_DIR = "pdb_cif_files"
VISUALIZATION_OUTPUT_DIR = os.path.join(
    BASE_RESULTS_DIR, "pymol_sessions_api"
)  # Новая папка

# Настройки режима визуализации: 'sample', 'list', 'all'
VISUALIZE_MODE = "sample"
SAMPLE_SIZE = 5  # Уменьшим для API, т.к. может быть медленнее
PDB_ID_LIST_TO_VISUALIZE = ["4XB4", "5FCX", "6ZLY"]


def format_pymol_selection(neighbors_list):
    """Преобразует список соседей в строку выбора PyMOL."""
    if not neighbors_list:
        return "none"  # PyMOL требует 'none', если выбор пуст
    # Формат: /object//chain/residue`icode
    selections = []
    for chain, _, resnum_icode in neighbors_list:
        # Для API лучше использовать полный путь к остатку
        # но для простоты пока оставим формат для select
        selections.append(f"(chain {chain} and resi {resnum_icode})")
    return " or ".join(selections)


def create_pymol_session_api(
    pdb_id,
    ligand_code,
    ring_neighbors_str,
    full_neighbors_str,
    cif_path,
    pse_output_path,
):
    """Создает PyMOL сессию с использованием API."""
    if not PYMOL_AVAILABLE:
        return False

    try:
        # 1. Очистить PyMOL перед загрузкой новой структуры
        cmd.reinitialize("everything")
        time.sleep(0.1)  # Небольшая пауза

        # 2. Загрузить структуру
        # print(f"  Загрузка: {cif_path}")
        cmd.load(cif_path, "main_structure")

        # 3. Базовая настройка вида
        cmd.remove("solvent")
        cmd.hide("everything", "all")
        cmd.show("cartoon", "polymer and not resn HOH+SO4+PO4+GOL+NAG")
        cmd.color("gray70", "polymer")
        cmd.show("sticks", f"resn {ligand_code}")
        cmd.color("magenta", f"resn {ligand_code}")
        cmd.util.cnc(f"resn {ligand_code}")

        # 4. Создание и раскраска выделений
        cmd.select("ring_neighbors", ring_neighbors_str)
        cmd.select("full_neighbors", full_neighbors_str)

        # print(f"  Соседи колец: {cmd.count_atoms('ring_neighbors')} атомов")
        # print(f"  Соседи всей цепи: {cmd.count_atoms('full_neighbors')} атомов")

        if cmd.count_atoms("full_neighbors") > 0:
            cmd.show("sticks", "full_neighbors")
            cmd.color("yellow", "full_neighbors")
        if cmd.count_atoms("ring_neighbors") > 0:
            cmd.show("sticks", "ring_neighbors")
            cmd.color("cyan", "ring_neighbors")  # Перекроет желтый

        # 5. Настройка камеры
        cmd.bg_color("white")
        cmd.center(f"resn {ligand_code}")
        zoom_sele = f"(resn {ligand_code}) or (vis and bound_to (resn {ligand_code}))"
        if cmd.count_atoms(zoom_sele) > 0:
            cmd.zoom(zoom_sele, buffer=4.0)
        else:  # Если лиганда или видимых соседей нет, зумим на все
            cmd.zoom()

        # 6. Сохранение сессии
        # print(f"  Сохранение в: {pse_output_path}")
        # Убедимся, что директория существует
        os.makedirs(os.path.dirname(pse_output_path), exist_ok=True)
        save_status = cmd.save(pse_output_path, format="pse")
        time.sleep(0.2)  # Пауза после сохранения

        # Проверка статуса сохранения (0 - успех, -1 - ошибка)
        if save_status == -1:
            print(f"! Ошибка сохранения PyMOL сессии для {pdb_id} в {pse_output_path}")
            return False

        # Дополнительная проверка файла (на всякий случай)
        if not os.path.exists(pse_output_path):
            print(
                f"! Файл сессии не найден после сохранения для {pdb_id}: {pse_output_path}"
            )
            return False

        return True

    except Exception as e:
        print(f"! Исключение при работе с PyMOL API для {pdb_id}: {e}")
        # traceback.print_exc() # Раскомментировать для отладки
        return False
    finally:
        # Очистка выделений и объектов для следующей итерации
        try:
            cmd.delete("all")
        except Exception:
            pass  # Игнорируем ошибки при очистке, если PyMOL уже недоступен


# --- Основной блок ---
if __name__ == "__main__":
    print("--- Создание PyMOL сессий v7 (Используя PyMOL API) ---")

    if not PYMOL_AVAILABLE:
        sys.exit(1)  # Выход, если PyMOL не импортировался

    # ... (Чтение и группировка данных - без изменений) ...
    ring_csv_path = os.path.join(RING_DIR, RAW_DATA_NAME)
    full_chain_csv_path = os.path.join(FULL_CHAIN_DIR, RAW_DATA_NAME)
    if not os.path.exists(ring_csv_path):
        exit(f"! Ошибка: Не найден файл {ring_csv_path}")
    if not os.path.exists(full_chain_csv_path):
        exit(f"! Ошибка: Не найден файл {full_chain_csv_path}")
    if not os.path.exists(CIF_DIR):
        exit(f"! Ошибка: Не найдена папка {CIF_DIR}")
    print("Чтение данных о соседях...")
    try:
        df_ring = pd.read_csv(ring_csv_path, dtype=str)
        df_full = pd.read_csv(full_chain_csv_path, dtype=str)
    except Exception as e:
        exit(f"! Ошибка чтения CSV файлов: {e}")
    print("Группировка данных...")
    try:
        ring_neighbors_dict = (
            df_ring.groupby("pdbID")
            .apply(
                lambda x: list(
                    zip(
                        x["Neighbor_Chain"], x["Neighbor_Resname"], x["Neighbor_Resnum"]
                    )
                ),
                include_groups=False,
            )
            .to_dict()
        )
        full_neighbors_dict = (
            df_full.groupby("pdbID")
            .apply(
                lambda x: list(
                    zip(
                        x["Neighbor_Chain"], x["Neighbor_Resname"], x["Neighbor_Resnum"]
                    )
                ),
                include_groups=False,
            )
            .to_dict()
        )
    except TypeError:
        ring_neighbors_dict = (
            df_ring.groupby("pdbID")
            .apply(
                lambda x: list(
                    zip(
                        x["Neighbor_Chain"], x["Neighbor_Resname"], x["Neighbor_Resnum"]
                    )
                )
            )
            .to_dict()
        )
        full_neighbors_dict = (
            df_full.groupby("pdbID")
            .apply(
                lambda x: list(
                    zip(
                        x["Neighbor_Chain"], x["Neighbor_Resname"], x["Neighbor_Resnum"]
                    )
                )
            )
            .to_dict()
        )
    pdb_ligand_map = (
        df_full.drop_duplicates("pdbID").set_index("pdbID")["Ligand"].to_dict()
    )

    # ... (Определение PDB ID для визуализации - без изменений) ...
    all_available_pdb_ids = sorted(list(pdb_ligand_map.keys()))
    pdb_ids_to_visualize = []
    print(f"\nВыбран режим визуализации: '{VISUALIZE_MODE}'")
    if VISUALIZE_MODE == "all":
        pdb_ids_to_visualize = all_available_pdb_ids
        print(f"Будет попытка создать {len(pdb_ids_to_visualize)} сессий.")
    elif VISUALIZE_MODE == "list":
        pdb_ids_to_visualize = [
            p for p in PDB_ID_LIST_TO_VISUALIZE if p in pdb_ligand_map
        ]
        not_found = [p for p in PDB_ID_LIST_TO_VISUALIZE if p not in pdb_ligand_map]
        if not_found:
            print(f"! Предупреждение: PDB ID не найдены: {', '.join(not_found)}")
        if not pdb_ids_to_visualize:
            exit("! Нет PDB ID из списка для визуализации.")
        print(
            f"Будет создано {len(pdb_ids_to_visualize)} сессий для: {', '.join(pdb_ids_to_visualize)}"
        )
    elif VISUALIZE_MODE == "sample":
        if not all_available_pdb_ids:
            exit("! Нет PDB ID для выборки.")
        actual_sample_size = min(SAMPLE_SIZE, len(all_available_pdb_ids))
        pdb_ids_to_visualize = random.sample(all_available_pdb_ids, actual_sample_size)
        print(
            f"Будет создано {len(pdb_ids_to_visualize)} сессий для случайной выборки."
        )
    else:
        exit(f"! Неизвестный режим: '{VISUALIZE_MODE}'.")

    # --- Создание сессий ---
    print(f"\nСоздание PyMOL сессий в '{VISUALIZATION_OUTPUT_DIR}'...")
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    success_viz_count = 0
    fail_viz_count = 0
    print(f"Начинается обработка {len(pdb_ids_to_visualize)} PDB ID...")
    # Цикл без tqdm, т.к. API может быть медленным и вывод важен
    for i, pdb_id in enumerate(pdb_ids_to_visualize):
        print(f"--- Обработка {i+1}/{len(pdb_ids_to_visualize)}: {pdb_id} ---")
        cif_path = os.path.join(CIF_DIR, f"{pdb_id}.cif")
        if not os.path.exists(cif_path):
            print(f"! Пропуск: CIF файл не найден {cif_path}")
            fail_viz_count += 1
            continue

        ring_neighbors = ring_neighbors_dict.get(pdb_id, [])
        full_neighbors = full_neighbors_dict.get(pdb_id, [])
        ligand_code = pdb_ligand_map.get(pdb_id)
        if not ligand_code:
            print(f"! Пропуск: Лиганд не найден для {pdb_id}")
            fail_viz_count += 1
            continue

        ring_sel_str = format_pymol_selection(ring_neighbors)
        full_sel_str = format_pymol_selection(full_neighbors)
        pse_output_path = os.path.join(
            VISUALIZATION_OUTPUT_DIR, f"{pdb_id}_{ligand_code}_neighbors.pse"
        )
        print(f"  Целевой файл: {pse_output_path}")

        # Вызываем новую функцию
        if create_pymol_session_api(
            pdb_id,
            ligand_code,
            ring_sel_str,
            full_sel_str,
            cif_path,
            pse_output_path,
        ):
            success_viz_count += 1
            print(f"  Сессия для {pdb_id} успешно создана.")
        else:
            fail_viz_count += 1
            print(f"  Не удалось создать сессию для {pdb_id}.")
            # Файл не должен был создаться, если функция вернула False

    print("\n--- Результаты создания визуализаций ---")
    print(f"Успешно создано сессий: {success_viz_count}")
    print(f"Ошибок при создании:   {fail_viz_count}")
    print(f"\nГотовые .pse файлы сохранены в: {VISUALIZATION_OUTPUT_DIR}")

    # Важно: Завершить PyMOL после использования API
    if PYMOL_AVAILABLE:
        try:
            # pymol.quit() # Может завершать весь скрипт
            pass  # Обычно не требуется явный quit при использовании как модуля
        except Exception:
            pass
