# -*- coding: utf-8 -*-
"""
Скрипт для анализа окружения КОЛЕЦ И ИХ ЗАМЕСТИТЕЛЕЙ каротиноидов,
с автоматическим поиском колец (NetworkX) и параллельной обработкой.
Включает расширенные опции визуализации (Heatmap, UMAP).
"""

import os
import pandas as pd
import warnings
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import concurrent.futures
import time

# --- Графовая библиотека ---
try:
    import networkx as nx
except ImportError:
    print("Ошибка: Библиотека NetworkX не найдена.")
    print("Пожалуйста, установите ее: pip install networkx")
    exit()

# --- Библиотеки для работы с PDB ---
try:
    from Bio.PDB import MMCIFParser, NeighborSearch
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
except ImportError:
    print("Ошибка: Библиотека Biopython не найдена.")
    print("Пожалуйста, установите ее: pip install biopython")
    exit()

# --- Библиотеки для UMAP ---
try:
    import umap

    HAS_UMAP = True
except ImportError:
    print("Предупреждение: Библиотека umap-learn не найдена.")
    print("Визуализация UMAP будет недоступна.")
    print("Установите ее: pip install umap-learn")
    HAS_UMAP = False


# --- Конфигурация ---
CSV_FILE = "carotenoid_structures.csv"
CIF_DIR = "pdb_cif_files"
N_CORES = 16  # Количество ядер для параллельной обработки
DISTANCE_CUTOFF_NEIGHBORS = 4.5  # Расстояние для поиска АМИНОКИСЛОТНЫХ соседей (Å)
BOND_DISTANCE_CUTOFF = 1.7  # Макс. расстояние для ковалентной связи внутри лиганда (Å) - увеличено для надежности
RING_SIZE = 6  # Искать кольца именно такого размера
OUTPUT_PLOT_FREQ = f"ring{RING_SIZE}_subst_aa_freq.png"
OUTPUT_PLOT_HEATMAP = f"ring{RING_SIZE}_subst_aa_heatmap.png"
OUTPUT_PLOT_UMAP = f"ring{RING_SIZE}_subst_aa_umap.png"
PROCESS_LIMIT = None  # Установите число (e.g., 20) для быстрой проверки

# --- Подавление предупреждений ---
warnings.simplefilter("ignore", PDBConstructionWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
# np.seterr(invalid='ignore') # Игнорировать ошибки деления на ноль при нормализации (если будут пустые окружения)

# --- Стандартные 20 аминокислот ---
STANDARD_AMINO_ACIDS = [  # Список важен для порядка в векторах
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_INDEX_MAP = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}


# --- Функции Анализа ---


def build_ligand_graph(residue: Residue, bond_cutoff: float = BOND_DISTANCE_CUTOFF):
    """Строит граф связей для тяжелых атомов остатка лиганда."""
    graph = nx.Graph()
    # Используем serial_number как уникальный идентификатор атома в рамках PDB,
    # чтобы избежать проблем с хешированием объектов Atom между процессами
    heavy_atoms = {
        atom.serial_number: atom for atom in residue.get_atoms() if atom.element != "H"
    }

    if len(heavy_atoms) < 3:
        return graph, heavy_atoms  # Возвращаем пустой граф и словарь атомов

    # Добавляем узлы
    graph.add_nodes_from(heavy_atoms.keys())

    # Добавляем ребра на основе расстояний
    atom_list = list(heavy_atoms.values())
    coords = np.array([atom.get_coord() for atom in atom_list])
    num_atoms = len(atom_list)

    try:
        from scipy.spatial.distance import pdist, squareform

        dist_matrix = squareform(pdist(coords))
    except ImportError:
        dist_matrix = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                # Проверка на NaN в координатах
                if np.isnan(coords[i]).any() or np.isnan(coords[j]).any():
                    dist = np.inf  # Не создаем связь, если координаты плохие
                else:
                    dist = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist

    epsilon = 1e-6
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = dist_matrix[i, j]
            if epsilon < dist <= bond_cutoff:
                # Добавляем ребро между serial_number атомов
                graph.add_edge(atom_list[i].serial_number, atom_list[j].serial_number)

    return graph, heavy_atoms


def find_ring_and_substituent_atoms(
    residue: Residue,
    ring_size: int = RING_SIZE,
    bond_cutoff: float = BOND_DISTANCE_CUTOFF,
):
    """
    Находит атомы колец заданного размера и их прямых соседей (заместителей).
    Возвращает множество объектов Atom.
    """
    graph, heavy_atoms_dict = build_ligand_graph(residue, bond_cutoff)
    if not graph or not heavy_atoms_dict:
        return set()

    ring_atom_serials = set()
    all_target_atoms = set()

    try:
        # Используем поиск по базису циклов - часто стабильнее для биомолекул
        cycles_basis = nx.cycle_basis(graph)
        target_cycles = [cycle for cycle in cycles_basis if len(cycle) == ring_size]

        # Если cycle_basis не нашел, пробуем simple_cycles (может быть дольше)
        if not target_cycles:
            all_cycles = list(nx.simple_cycles(graph))
            target_cycles = [cycle for cycle in all_cycles if len(cycle) == ring_size]

        # Собираем serial_number атомов из найденных колец
        for cycle in target_cycles:
            ring_atom_serials.update(cycle)

        if not ring_atom_serials:
            # print(f"  Info: Не найдено {ring_size}-членных колец для {residue.get_resname()} {residue.id}")
            return set()  # Колец нет, возвращаем пустое множество

        # Находим заместители
        substituent_atom_serials = set()
        for ring_atom_serial in ring_atom_serials:
            # Ищем соседей в графе по serial_number
            neighbors = graph.neighbors(ring_atom_serial)
            for neighbor_serial in neighbors:
                # Если сосед не является частью найденных колец, это заместитель
                if neighbor_serial not in ring_atom_serials:
                    substituent_atom_serials.add(neighbor_serial)

        # Объединяем serial_number атомов колец и заместителей
        all_target_serials = ring_atom_serials.union(substituent_atom_serials)

        # Преобразуем serial_number обратно в объекты Atom
        all_target_atoms = {
            heavy_atoms_dict[sn] for sn in all_target_serials if sn in heavy_atoms_dict
        }

    except Exception as e:
        print(
            f"  Warning: Ошибка при поиске циклов/заместителей для {residue.get_resname()} {residue.id}: {e}"
        )
        return set()

    return all_target_atoms


# --- Функция для обработки одной структуры (для параллелизации) ---


def process_structure(args):
    """
    Обрабатывает один CIF файл: находит лиганд, его кольца+заместители,
    и соседние аминокислоты белка.
    """
    pdb_id, ligand_code, cif_path, neighbor_distance_cutoff, ring_size, bond_cutoff = (
        args
    )
    # print(f"Processing {pdb_id}...") # Отладка

    parser = MMCIFParser(QUIET=True)
    try:
        if not os.path.exists(cif_path):
            # print(f"File not found: {cif_path}")
            return (
                pdb_id,
                ligand_code,
                None,
                "File not found",
            )  # Возвращаем None вместо списка соседей
        structure = parser.get_structure(pdb_id, cif_path)
    except Exception as e:
        # print(f"Error parsing {pdb_id}: {e}")
        return pdb_id, ligand_code, None, f"Parsing error: {e}"

    target_atoms_for_search = set()  # Множество объектов Atom колец+заместителей
    protein_atoms = []  # Список объектов Atom белка
    ligand_residues_found = []  # Список объектов Residue лиганда

    try:
        if not structure.child_list:
            return pdb_id, ligand_code, None, "No models in structure"
        model = structure[0]

        # Собираем атомы и определяем атомы колец+заместителей
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith("H_") and residue.resname == ligand_code:
                    ligand_residues_found.append(residue)
                    current_target_atoms = find_ring_and_substituent_atoms(
                        residue, ring_size=ring_size, bond_cutoff=bond_cutoff
                    )
                    if current_target_atoms:
                        target_atoms_for_search.update(current_target_atoms)

                elif (
                    residue.id[0] == " " and residue.resname in AA_INDEX_MAP
                ):  # Проверяем по словарю для скорости
                    for atom in residue:
                        # Проверяем координаты перед добавлением
                        if atom.coord is not None and not np.isnan(atom.coord).any():
                            protein_atoms.append(atom)

        # Проверки
        if not ligand_residues_found:
            return pdb_id, ligand_code, [], "Ligand not found"  # Пустой список соседей

        if not target_atoms_for_search:
            # print(f"  Warning: Лиганд {ligand_code} найден в {pdb_id}, но не найдены атомы колец({ring_size})+заместителей.")
            return (
                pdb_id,
                ligand_code,
                [],
                "Ring/Substituent atoms not found",
            )  # Пустой список

        if not protein_atoms:
            # print(f"  Warning: Не найдено атомов белка в файле {pdb_id}")
            return pdb_id, ligand_code, [], "No protein atoms found"  # Пустой список

        # Поиск соседей
        ns = NeighborSearch(protein_atoms)
        neighbor_atoms = set()
        for target_atom in target_atoms_for_search:
            try:
                # Еще раз проверим координаты атома перед поиском
                if target_atom.coord is None or np.isnan(target_atom.coord).any():
                    continue
                nearby_atoms = ns.search(
                    target_atom.coord, neighbor_distance_cutoff, level="A"
                )
                neighbor_atoms.update(nearby_atoms)
            except Exception as e:
                # print(f"Error during neighbor search for atom {target_atom} in {pdb_id}: {e}")
                continue  # Пропускаем этот атом

        # Получаем уникальные родительские остатки
        neighbor_residues = set()
        for atom in neighbor_atoms:
            residue = atom.get_parent()
            # Дополнительная проверка на стандартность
            if (
                isinstance(residue, Residue)
                and residue.id[0] == " "
                and residue.resname in AA_INDEX_MAP
            ):
                neighbor_residues.add(residue)

        # Форматируем результат
        neighbor_residue_info = []
        for res in neighbor_residues:
            chain_id = res.get_parent().id
            resnum = res.id[1]
            insertion_code = res.id[2].strip()
            res_identifier = (
                f"{resnum}{insertion_code}" if insertion_code else str(resnum)
            )
            neighbor_residue_info.append((chain_id, res.resname, res_identifier))

        neighbor_residue_info.sort()
        # print(f"Finished {pdb_id} - {len(neighbor_residue_info)} neighbors")
        return (
            pdb_id,
            ligand_code,
            neighbor_residue_info,
            "Success",
        )  # Возвращаем список соседей

    except Exception as e:
        # Общий обработчик ошибок на случай непредвиденного
        # print(f"Unexpected error processing {pdb_id}: {e}")
        # traceback.print_exc() # Раскомментировать для детальной отладки
        return pdb_id, ligand_code, None, f"Unexpected error: {e}"


# --- Функции Визуализации ---


def plot_amino_acid_frequencies(
    aa_counts, filename="aa_frequency_plot.png", title_suffix=""
):
    """Строит и сохраняет гистограмму частот аминокислот."""
    if not aa_counts:
        print("Нет данных для построения графика частот.")
        return

    # Убедимся, что считаем только стандартные АА для графика
    standard_aa_counts = {
        aa: count for aa, count in aa_counts.items() if aa in AA_INDEX_MAP
    }
    if not standard_aa_counts:
        print("Не найдено стандартных аминокислот для построения графика частот.")
        return

    sorted_counts = dict(
        sorted(standard_aa_counts.items(), key=lambda item: item[1], reverse=True)
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x=list(sorted_counts.keys()),
        y=list(sorted_counts.values()),
        palette="viridis",
        order=STANDARD_AMINO_ACIDS,
    )
    plt.xlabel("Аминокислота", fontsize=12)
    plt.ylabel(f"Частота встречаемости {title_suffix}", fontsize=12)
    plt.title(
        f"Частота аминокислот в окружении ({DISTANCE_CUTOFF_NEIGHBORS} Å) {RING_SIZE}-чл. КОЛЕЦ+ЗАМЕСТИТЕЛЕЙ",
        fontsize=14,
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300)
        print(f"\nГистограмма частот сохранена в файл: {filename}")
        plt.close()  # Закрываем фигуру после сохранения
    except Exception as e:
        print(f"\nОшибка сохранения графика частот: {e}")


def plot_aa_heatmap(data_matrix, index_labels, filename="aa_heatmap.png"):
    """Строит тепловую карту частот аминокислот."""
    if data_matrix is None or data_matrix.shape[0] == 0:
        print("Нет данных для построения тепловой карты.")
        return

    plt.figure(
        figsize=(10, max(8, data_matrix.shape[0] * 0.5))
    )  # Адаптивный размер по высоте
    sns.heatmap(
        data_matrix,
        annot=False,
        cmap="viridis",
        linewidths=0.5,
        xticklabels=STANDARD_AMINO_ACIDS,
        yticklabels=index_labels,
    )
    plt.xlabel("Аминокислота в окружении", fontsize=12)
    plt.ylabel("Структура (PDB ID)", fontsize=12)
    plt.title(
        "Тепловая карта нормализованных частот аминокислот в окружении", fontsize=14
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300)
        print(f"Тепловая карта сохранена в файл: {filename}")
        plt.close()
    except Exception as e:
        print(f"\nОшибка сохранения тепловой карты: {e}")


def plot_umap_results(
    embedding,
    labels,
    color_map,
    filename="umap_plot.png",
    title="UMAP проекция окружения",
):
    """Строит UMAP-график."""
    if embedding is None or embedding.shape[0] == 0:
        print("Нет данных для построения UMAP графика.")
        return

    plt.figure(figsize=(12, 10))
    unique_labels = sorted(list(set(labels)))
    # Создаем цветовую палитру
    lut = dict(zip(unique_labels, sns.color_palette("husl", len(unique_labels))))
    colors = [lut[l] for l in labels]

    plt.scatter(
        embedding[:, 0], embedding[:, 1], c=colors, s=20, alpha=0.7
    )  # s - размер точки

    # Создаем легенду вручную
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=lut[label],
            markersize=10,
        )
        for label in unique_labels
    ]
    plt.legend(
        handles=handles, title=color_map, bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Оставляем место для легенды

    try:
        plt.savefig(filename, dpi=300)
        print(f"UMAP график сохранен в файл: {filename}")
        plt.close()
    except Exception as e:
        print(f"\nОшибка сохранения UMAP графика: {e}")


# --- Основной блок выполнения ---
if __name__ == "__main__":
    main_start_time = time.time()
    print(
        f"--- Анализ окружения {RING_SIZE}-членных КОЛЕЦ+ЗАМЕСТИТЕЛЕЙ (Параллельно, {N_CORES} ядер) ---"
    )
    print(f"Чтение метаданных из: {CSV_FILE}")
    # ... (проверки наличия CSV и папки CIF) ...
    if not os.path.exists(CSV_FILE):
        exit(f"Ошибка: Файл {CSV_FILE} не найден!")
    if not os.path.exists(CIF_DIR):
        exit(f"Ошибка: Папка {CIF_DIR} не найдена!")

    try:
        df = pd.read_csv(CSV_FILE, dtype={"pdbID": str})
        df["pdbID"] = df["pdbID"].str.strip()
        print(f"Загружено {len(df)} записей.")
    except Exception as e:
        exit(f"Ошибка чтения CSV файла: {e}")

    # --- Подготовка задач для параллельной обработки ---
    tasks = []
    pdb_ids_to_process = df["pdbID"].unique()  # Обрабатываем каждый PDB ID один раз
    if PROCESS_LIMIT:
        pdb_ids_to_process = pdb_ids_to_process[:PROCESS_LIMIT]
        print(f"Ограничение обработки: первые {PROCESS_LIMIT} уникальных PDB ID.")

    print(f"Подготовка задач для {len(pdb_ids_to_process)} уникальных PDB ID...")
    for pdb_id in pdb_ids_to_process:
        # Находим все строки в df для этого PDB ID, чтобы знать все лиганды
        relevant_rows = df[df["pdbID"] == pdb_id]
        if relevant_rows.empty:
            continue

        # Используем первый лиганд из списка для этого PDB (можно усложнить логику, если нужно)
        ligand_code = relevant_rows["Ligand"].iloc[0]
        cif_path = os.path.join(CIF_DIR, f"{pdb_id}.cif")
        if os.path.exists(cif_path):
            tasks.append(
                (
                    pdb_id,
                    ligand_code,
                    cif_path,
                    DISTANCE_CUTOFF_NEIGHBORS,
                    RING_SIZE,
                    BOND_DISTANCE_CUTOFF,
                )
            )
        else:
            print(f"Предупреждение: Пропуск {pdb_id}, т.к. файл {cif_path} не найден.")

    if not tasks:
        exit("Нет задач для обработки (проверьте наличие CIF файлов).")

    print(f"\nЗапуск параллельной обработки ({len(tasks)} задач) на {N_CORES} ядрах...")
    results_list = (
        []
    )  # Список для хранения всех соседних остатков [(pdb_id, ligand, [(chain, resname, num),...]), ...]
    success_count = 0
    fail_count = 0
    processed_results = {}  # Словарь {pdb_id: [ (chain, resname, num), ... ]}

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_CORES) as executor:
        # Используем map для сохранения порядка (если важно) или submit/as_completed для прогресса
        # future_to_task = {executor.submit(process_structure, task): task for task in tasks}
        # for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Обработка PDB"):
        #     task_info = future_to_task[future]
        #     pdb_id = task_info[0]
        #     try:
        #         pdb_id_res, ligand_res, neighbors, status = future.result()
        #         if neighbors is not None: # Успешно или пустой список соседей
        #             processed_results[pdb_id_res] = neighbors
        #             results_list.append((pdb_id_res, ligand_res, neighbors)) # Сохраняем для детального анализа, если нужно
        #             success_count += 1
        #         else: # Ошибка обработки (neighbors is None)
        #             print(f"  Ошибка для {pdb_id_res}: {status}")
        #             fail_count += 1
        #     except Exception as exc:
        #         print(f'Задача для {pdb_id} вызвала исключение: {exc}')
        #         fail_count += 1

        # Альтернатива с map - проще, но без прогресс-бара по задачам
        results_iterator = executor.map(process_structure, tasks)
        for i, result in enumerate(
            tqdm(results_iterator, total=len(tasks), desc="Обработка PDB")
        ):
            pdb_id_res, ligand_res, neighbors, status = result
            if neighbors is not None:  # Успешно или пустой список соседей
                processed_results[pdb_id_res] = neighbors
                # results_list.append((pdb_id_res, ligand_res, neighbors)) # Можно раскомментировать, если нужны детали
                if status == "Success" or isinstance(
                    neighbors, list
                ):  # Считаем успехом, даже если соседей 0
                    success_count += 1
                else:  # Статус ошибки, но neighbors не None (например, [])
                    # print(f"  Обработано {pdb_id_res} со статусом: {status}")
                    success_count += (
                        1  # Считаем за успех, если вернулся список (пусть и пустой)
                    )
            else:  # Ошибка обработки (neighbors is None)
                print(f"  Ошибка для {pdb_id_res}: {status}")
                fail_count += 1

    print("\n--- Статистика Параллельной Обработки ---")
    print(f"Успешно обработано PDB ID (или вернулся пустой список): {success_count}")
    print(f"Ошибки чтения/парсинга/анализа:                       {fail_count}")

    if not processed_results:
        exit("Нет успешно обработанных данных для анализа и визуализации.")

    # --- Агрегация данных для анализа ---
    print("\nАгрегация результатов...")
    all_neighbor_residues_info = []
    structure_aa_vectors = {}  # {pdb_id: np.array([freq_ala, freq_arg,...])}
    structure_labels = {}  # {pdb_id: ligand_code}

    for pdb_id, neighbors in processed_results.items():
        all_neighbor_residues_info.extend(neighbors)  # Для общей гистограммы частот

        # Создаем вектор частот для данной структуры
        aa_vector = np.zeros(len(STANDARD_AMINO_ACIDS))
        if neighbors:
            counts = Counter(
                resname for _, resname, _ in neighbors if resname in AA_INDEX_MAP
            )
            total_neighbors = len(
                neighbors
            )  # Сумма всех соседей (не только стандартных, если попали)
            if total_neighbors > 0:
                for aa, index in AA_INDEX_MAP.items():
                    aa_vector[index] = counts.get(
                        aa, 0
                    )  # / total_neighbors # Нормализация? Пока оставим абсолютные числа
        structure_aa_vectors[pdb_id] = aa_vector
        # Получаем лиганд из исходного DataFrame для метки
        ligand = df[df["pdbID"] == pdb_id]["Ligand"].iloc[0]
        structure_labels[pdb_id] = ligand

    print(
        f"Всего уникальных остатков в окружении колец+зам.: {len(all_neighbor_residues_info)}"
    )

    # --- Общая Гистограмма Частот ---
    if all_neighbor_residues_info:
        overall_aa_counts = Counter(
            resname for _, resname, _ in all_neighbor_residues_info
        )
        plot_amino_acid_frequencies(overall_aa_counts, filename=OUTPUT_PLOT_FREQ)
    else:
        print("Нет данных для общей гистограммы частот.")

    # --- Подготовка данных для Heatmap и UMAP ---
    pdb_order = sorted(structure_aa_vectors.keys())  # Фиксируем порядок PDB ID
    if not pdb_order:
        exit("Нет данных для Heatmap и UMAP.")

    data_matrix = np.array([structure_aa_vectors[pdb_id] for pdb_id in pdb_order])
    labels_list = [structure_labels[pdb_id] for pdb_id in pdb_order]

    # Нормализация данных (например, по строкам - профиль каждой структуры)
    # Делаем это аккуратно, чтобы избежать деления на ноль, если все частоты нулевые
    row_sums = data_matrix.sum(axis=1, keepdims=True)
    # Заменяем 0 на 1 в суммах, чтобы избежать деления на ноль (там, где сумма 0, частоты останутся 0)
    safe_row_sums = np.where(row_sums == 0, 1, row_sums)
    normalized_data_matrix = data_matrix / safe_row_sums

    # --- Тепловая карта ---
    plot_aa_heatmap(
        normalized_data_matrix, index_labels=pdb_order, filename=OUTPUT_PLOT_HEATMAP
    )

    # --- UMAP Визуализация ---
    if HAS_UMAP and data_matrix.shape[0] > 1 and data_matrix.shape[1] > 1:
        print("\nЗапуск UMAP...")
        try:
            reducer = umap.UMAP(
                n_neighbors=min(
                    15, data_matrix.shape[0] - 1
                ),  # Важно: n_neighbors < samples
                min_dist=0.1,
                n_components=2,
                metric="hellinger",  # Метрика Хеллингера хорошо подходит для частот/гистограмм
                # metric='cosine', # Альтернатива
                random_state=42,
            )
            embedding = reducer.fit_transform(
                data_matrix
            )  # Используем НЕ нормализованные данные, т.к. Hellinger работает с ними

            plot_umap_results(
                embedding,
                labels=labels_list,
                color_map="Ligand",
                filename=OUTPUT_PLOT_UMAP,
                title="UMAP проекция окружения (раскраска по лиганду)",
            )
        except Exception as e:
            print(f"Ошибка выполнения UMAP: {e}")
            # traceback.print_exc()
    elif not HAS_UMAP:
        print("\nUMAP визуализация пропущена (библиотека umap-learn не найдена).")
    else:
        print("\nUMAP визуализация пропущена (недостаточно данных).")

    # --- Завершение ---
    main_end_time = time.time()
    print(
        f"\n--- Анализ и Визуализация завершены за {main_end_time - main_start_time:.2f} секунд ---"
    )
