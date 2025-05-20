# -*- coding: utf-8 -*-
"""
Скрипт для анализа геометрических параметров взаимодействий
между каротиноидами (иононовое кольцо) и ароматическими аминокислотами.

Рассчитывает:
- Расстояние между центроидами колец.
- Угол между нормалями к плоскостям колец.

Визуализирует распределения этих метрик.
Добавлено кэширование рассчитанной геометрии.
Добавлена нормировка для faceted-гистограммы углов.
"""

import concurrent.futures
import math
import os
import time
import traceback
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import MMCIFParser, PDBExceptions
from Bio.PDB.vectors import calc_dihedral, Vector
from tqdm import tqdm

# --- Испорты, необходимые для build_ligand_graph ---
try:
    import networkx as nx
except ImportError:
    exit("! Ошибка: NetworkX не найден. pip install networkx")
try:
    from scipy.spatial.distance import pdist, squareform

    HAS_SCIPY = True
except ImportError:
    # print("! Предупреждение: SciPy не найден. Расчет связей в build_ligand_graph будет медленнее.")
    HAS_SCIPY = False
# --------------------------------------------------


# --- Конфигурация ---
# Папки с результатами предыдущего скрипта (v6)
BASE_RESULTS_DIR = "analysis_results_v6_enriched"
# Выбираем, какие соседи анализировать (окружение колец кажется релевантнее)
NEIGHBOR_DATA_DIR = os.path.join(BASE_RESULTS_DIR, "ring_subst_env")
NEIGHBOR_CSV_FILE = os.path.join(NEIGHBOR_DATA_DIR, "neighbor_residues_raw.csv")
METADATA_CSV_FILE = "carotenoid_structures.csv"  # Из parse.py
CIF_DIR = "pdb_cif_files"
OUTPUT_DIR = os.path.join(
    BASE_RESULTS_DIR, "geometry_analysis"
)  # Новая папка для результатов геометрии
GEOMETRY_CSV_FILE = os.path.join(OUTPUT_DIR, "aromatic_ionone_geometry.csv")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# --- Параметры Анализа ---
AROMATIC_AA = ["PHE", "TYR", "TRP"]
# Атомы, формирующие ароматические кольца
AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],  # Без OH
    "TRP": ["CG", "CD1", "NE1", "CE2", "CD2"],  # Только 5-членное кольцо (индол)
}
MAX_WORKERS = 16
# Макс. расстояние между центроидами колец для учета взаимодействия (Ангстрем)
MAX_CENTROID_DISTANCE = 7.0  # Оставляем 7.0 для начала
MIN_ATOMS_FOR_PLANE = 3  # Минимум атомов для определения плоскости
BOND_DISTANCE_CUTOFF = 1.9 # Для build_ligand_graph внутри get_ionone_ring_atoms

warnings.simplefilter("ignore", PDBExceptions.PDBConstructionWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter(
    "ignore", UserWarning
)  # Игнорируем UserWarning от Biopython calc_dihedral
np.set_printoptions(precision=3)


# --- Вспомогательные Функции ---


# Функция из analyze_pdb.py, нужна для get_ionone_ring_atoms
def build_ligand_graph(residue, bond_cutoff: float = BOND_DISTANCE_CUTOFF):
    """Строит граф связей для тяжелых атомов остатка."""
    graph = nx.Graph()
    heavy_atoms = {
        a.serial_number: a for a in residue.get_atoms() if a.element != "H"
    }
    if len(heavy_atoms) < 3:
        return graph, heavy_atoms
    graph.add_nodes_from(heavy_atoms.keys())
    atom_list = list(heavy_atoms.values())
    coords = np.array([a.get_coord() for a in atom_list])
    num_atoms = len(atom_list)
    valid_coords = ~np.isnan(coords).any(axis=1)
    if not np.all(valid_coords):
        original_indices = np.where(valid_coords)[0]
        atom_list = [atom_list[i] for i in original_indices]
        coords = coords[valid_coords]
        num_atoms = len(atom_list)
        if num_atoms < 2:
            return graph, heavy_atoms
    dist_matrix = None
    HAS_SCIPY_RUNTIME = HAS_SCIPY
    if HAS_SCIPY_RUNTIME:
        try:
            dist_matrix = squareform(pdist(coords))
        except Exception:
            HAS_SCIPY_RUNTIME = False
    if dist_matrix is None:
        dist_matrix = np.zeros((num_atoms, num_atoms))
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist_matrix[i, j] = dist_matrix[j, i] = np.linalg.norm(
                    coords[i] - coords[j]
                )
    epsilon = 1e-6
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if epsilon < dist_matrix[i, j] <= bond_cutoff:
                graph.add_edge(
                    atom_list[i].serial_number, atom_list[j].serial_number
                )
    return graph, heavy_atoms


def get_atoms_by_names(residue, atom_names):
    """Возвращает список объектов Atom из остатка по именам."""
    atoms = []
    present_atom_names = {a.get_id() for a in residue.get_atoms()}
    for name in atom_names:
        if name in present_atom_names:
            try:
                atoms.append(residue[name])
            except KeyError:
                pass
    return atoms


def calculate_centroid(atoms):
    """Рассчитывает геометрический центр (центроид) для списка атомов."""
    if not atoms:
        return None
    coords = np.array([a.get_coord() for a in atoms if a.coord is not None])
    if coords.shape[0] < 1:
        return None
    return np.mean(coords, axis=0)


def calculate_plane_normal(atoms):
    """
    Рассчитывает нормаль к плоскости, определенной атомами, используя PCA (SVD).
    """
    if len(atoms) < MIN_ATOMS_FOR_PLANE:
        return None
    coords = np.array([a.get_coord() for a in atoms if a.coord is not None])
    if coords.shape[0] < MIN_ATOMS_FOR_PLANE:
        return None
    center = np.mean(coords, axis=0)
    centered_coords = coords - center
    try:
        _, _, vh = np.linalg.svd(centered_coords)
        normal = vh[-1]
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-6: # Проверка на нулевую нормаль
            return None
        return normal / norm_val
    except np.linalg.LinAlgError:
        return None


def calculate_vector_angle(v1, v2):
    """Рассчитывает угол (в градусах, 0-90) между двумя векторами."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return np.nan # Нельзя рассчитать угол с нулевым вектором
    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return min(angle_deg, 180.0 - angle_deg)


def find_ligand_residues(model, ligand_code):
    """Находит все остатки лиганда с заданным кодом в модели."""
    ligand_residues = []
    for chain in model:
        for residue in chain:
            if residue.id[0].startswith("H_") and residue.resname == ligand_code:
                ligand_residues.append(residue)
    return ligand_residues


def find_specific_neighbor(model, chain_id, resnum_icode):
    """Находит конкретный остаток соседа по ID цепи и номеру/iCode."""
    try:
        chain = model[chain_id]
        resnum_str = "".join(filter(str.isdigit, str(resnum_icode)))
        icode = "".join(filter(str.isalpha, str(resnum_icode)))
        if not icode:
            icode = " "

        residue_id = (" ", int(resnum_str), icode)
        if chain.has_id(residue_id):
            return chain[residue_id]
        else:
            residue_id_no_icode = (" ", int(resnum_str), " ")
            if chain.has_id(residue_id_no_icode):
                return chain[residue_id_no_icode]

    except KeyError:
        pass
    except ValueError:
        # print(f"  Предупреждение: Не удалось распарсить номер остатка: {resnum_icode}")
        pass
    return None


def get_ionone_ring_atoms(ligand_residue):
    """
    Находит атомы 6-членного(ых) кольца(ец) лиганда (предположительно иононовых).
    """
    found_rings_atoms = []
    try:
        graph, heavy_atoms_dict = build_ligand_graph(
            ligand_residue, bond_cutoff=BOND_DISTANCE_CUTOFF
        )
        if not graph or not heavy_atoms_dict:
            return []

        target_cycles_serials = []
        try: # Предпочитаем cycle_basis, если граф связный
            cycles = list(nx.cycle_basis(graph))
        except nx.NetworkXNotImplemented: # Для несвязных графов
            cycles = list(nx.simple_cycles(graph))

        target_cycles = [c for c in cycles if len(c) == 6]
        # Если cycle_basis не нашел 6-членные, пробуем simple_cycles
        if not target_cycles and not isinstance(cycles, list):
             cycles = list(nx.simple_cycles(graph))
             target_cycles = [c for c in cycles if len(c) == 6]


        for cycle_serials in target_cycles:
            ring_atoms = [
                heavy_atoms_dict[sn] for sn in cycle_serials if sn in heavy_atoms_dict
            ]
            if len(ring_atoms) == 6:
                carbon_count = sum(1 for a in ring_atoms if a.element == "C")
                if carbon_count >= 4:
                    found_rings_atoms.append(ring_atoms)

    except Exception as e:
        # print(f"  Ошибка при поиске колец лиганда {ligand_residue.get_full_id()}: {e}")
        return []
    return found_rings_atoms


# --- Основная функция обработки структуры ---


def process_structure_geometry(args):
    """
    Обрабатывает одну PDB структуру для расчета геометрии взаимодействий.
    """
    pdb_id, ligand_code, neighbors_info, cif_path = args
    results = []
    parser = MMCIFParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_id, cif_path)
        if not structure.child_list:
            return results
        model = structure[0]

        ligand_residues = find_ligand_residues(model, ligand_code)
        if not ligand_residues:
            return results

        aromatic_neighbors = [
            n for n in neighbors_info if n["resname"] in AROMATIC_AA
        ]
        if not aromatic_neighbors:
            return results

        ligand_instance_counter = 0
        for lig_res in ligand_residues:
            ligand_instance_counter += 1
            # Создаем уникальный ID для экземпляра лиганда (цепь + номер)
            ligand_id_str = f"{lig_res.get_parent().id}_{lig_res.id[1]}"
            if lig_res.id[2].strip(): # Добавляем iCode если есть
                 ligand_id_str += lig_res.id[2].strip()


            ionone_rings_atoms = get_ionone_ring_atoms(lig_res)
            if not ionone_rings_atoms:
                continue

            ionone_ring_counter = 0
            for ionone_atoms in ionone_rings_atoms:
                ionone_ring_counter += 1
                ionone_centroid = calculate_centroid(ionone_atoms)
                ionone_normal = calculate_plane_normal(ionone_atoms)

                if ionone_centroid is None or ionone_normal is None:
                    continue

                for neighbor in aromatic_neighbors:
                    neighbor_resname = neighbor["resname"]
                    neighbor_chain_id = neighbor["chain"]
                    neighbor_resid = neighbor["resnum_icode"]

                    aa_res = find_specific_neighbor(
                        model, neighbor_chain_id, neighbor_resid
                    )
                    if aa_res is None or aa_res.resname != neighbor_resname:
                        continue

                    ring_atom_names = AROMATIC_RING_ATOMS.get(neighbor_resname)
                    if not ring_atom_names:
                        continue

                    aa_ring_atoms = get_atoms_by_names(aa_res, ring_atom_names)
                    if len(aa_ring_atoms) < MIN_ATOMS_FOR_PLANE:
                        continue

                    aa_centroid = calculate_centroid(aa_ring_atoms)
                    aa_normal = calculate_plane_normal(aa_ring_atoms)

                    if aa_centroid is None or aa_normal is None:
                        continue

                    centroid_distance = np.linalg.norm(ionone_centroid - aa_centroid)

                    if centroid_distance <= MAX_CENTROID_DISTANCE:
                        inter_plane_angle = calculate_vector_angle(
                            ionone_normal, aa_normal
                        )
                        if np.isnan(inter_plane_angle): # Пропускаем если угол не рассчитался
                            continue

                        results.append(
                            {
                                "PDB_ID": pdb_id,
                                "Ligand_Code": ligand_code,
                                "Ligand_Instance": ligand_id_str,
                                "Ionone_Ring_Num": ionone_ring_counter,
                                "Neighbor_AA": f"{neighbor_resname}_{neighbor_chain_id}_{neighbor_resid}",
                                "AA_Type": neighbor_resname,
                                "Centroid_Distance": round(centroid_distance, 3),
                                "Interplane_Angle": round(inter_plane_angle, 2),
                            }
                        )

    except FileNotFoundError:
        print(f"! Ошибка: CIF файл не найден: {cif_path}")
    except Exception as e:
        print(f"! Ошибка обработки PDB {pdb_id}: {e}")
        # traceback.print_exc() # Раскомментировать для отладки

    return results


# --- Основной Блок ---
if __name__ == "__main__":
    main_start_time = time.time()
    print(
        "--- Анализ Геометрии Взаимодействий (Иононовое кольцо vs Ароматические АА) ---"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df_geometry = None  # Инициализируем DataFrame

    # --- ПРОВЕРКА КЭША ГЕОМЕТРИИ ---
    if os.path.exists(GEOMETRY_CSV_FILE):
        print(f"\nНайден файл с геометрией: {GEOMETRY_CSV_FILE}")
        try:
            # Указываем типы данных для колонок, которые могут вызвать проблемы
            dtype_spec = {
                 "PDB_ID": str,
                 "Ligand_Code": str,
                 "Ligand_Instance": str,
                 "Ionone_Ring_Num": int, # Или str, если могут быть нечисловые
                 "Neighbor_AA": str,
                 "AA_Type": str,
                 "Centroid_Distance": float,
                 "Interplane_Angle": float
            }
            df_geometry = pd.read_csv(GEOMETRY_CSV_FILE, dtype=dtype_spec)
            print(f"Загружено {len(df_geometry)} записей геометрии из кэша.")
        except Exception as e:
            print(f"! Ошибка чтения кэша геометрии: {e}. Будет выполнен пересчет.")
            df_geometry = None  # Сбрасываем, чтобы запустить расчет
    else:
        print("\nФайл с геометрией не найден, будет выполнен расчет.")
        df_geometry = None
    # --------------------------------

    if df_geometry is None:  # Запускаем расчет только если кэш не загружен
        # --- 1. Загрузка данных ---
        print("\n1. Загрузка метаданных и данных о соседях...")
        if not os.path.exists(METADATA_CSV_FILE):
            exit(f"! Ошибка: Файл метаданных не найден: {METADATA_CSV_FILE}")
        if not os.path.exists(NEIGHBOR_CSV_FILE):
            exit(f"! Ошибка: Файл соседей не найден: {NEIGHBOR_CSV_FILE}")
        if not os.path.exists(CIF_DIR):
            exit(f"! Ошибка: Папка с CIF файлами не найдена: {CIF_DIR}")

        try:
            df_meta = pd.read_csv(METADATA_CSV_FILE, dtype={"pdbID": str})
            pdb_ligand_map = df_meta.set_index("pdbID")["Ligand"].to_dict()

            df_neighbors = pd.read_csv(
                NEIGHBOR_CSV_FILE,
                dtype={
                    "pdbID": str,
                    "Ligand": str,
                    "Neighbor_Chain": str,
                    "Neighbor_Resname": str,
                    "Neighbor_Resnum": str,  # Важно читать как строку
                },
            )
            df_neighbors = df_neighbors.rename(
                columns={
                    "pdbID": "PDB_ID",
                    "Ligand": "Ligand_Code",
                    "Neighbor_Chain": "chain",
                    "Neighbor_Resname": "resname",
                    "Neighbor_Resnum": "resnum_icode",
                }
            )
            print(f"Загружено {len(df_neighbors)} записей о соседях.")
        except Exception as e:
            exit(f"! Ошибка загрузки CSV файлов: {e}")

        # Группируем соседей по PDB ID
        neighbors_by_pdb = defaultdict(list)
        required_cols = ["PDB_ID", "chain", "resname", "resnum_icode"]
        for idx, row in df_neighbors.iterrows():
            if not all(pd.notna(row[col]) for col in required_cols):
                continue
            neighbors_by_pdb[row["PDB_ID"]].append(row[required_cols].to_dict())

        # --- 2. Подготовка задач ---
        print("\n2. Подготовка задач для анализа геометрии...")
        tasks = []
        pdb_ids_with_neighbors = list(neighbors_by_pdb.keys())
        print(f"Всего PDB ID с соседями для анализа: {len(pdb_ids_with_neighbors)}")

        skipped_no_ligand = 0
        skipped_no_cif = 0
        for pdb_id in tqdm(pdb_ids_with_neighbors, desc="Подготовка задач"):
            ligand_code = pdb_ligand_map.get(pdb_id)
            if not ligand_code:
                skipped_no_ligand += 1
                continue
            cif_path = os.path.join(CIF_DIR, f"{pdb_id}.cif")
            if not os.path.exists(cif_path):
                skipped_no_cif += 1
                continue

            task_neighbors = neighbors_by_pdb[pdb_id]
            tasks.append((pdb_id, ligand_code, task_neighbors, cif_path))

        if skipped_no_ligand > 0:
            print(f"  Пропущено PDB ID без лиганда в метаданных: {skipped_no_ligand}")
        if skipped_no_cif > 0:
            print(f"  Пропущено PDB ID без CIF файла: {skipped_no_cif}")
        if not tasks:
            exit("! Нет задач для обработки.")
        print(f"Подготовлено задач для анализа: {len(tasks)}")

        # --- 3. Параллельный расчет геометрии ---
        print(
            f"\n3. Запуск расчета геометрии ({len(tasks)} задач) на {MAX_WORKERS} ядрах..."
        )
        all_geometry_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pdb = {
                executor.submit(process_structure_geometry, task): task[0] for task in tasks
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_pdb),
                total=len(tasks),
                desc="Расчет геометрии",
            ):
                pdb_id = future_to_pdb[future]
                try:
                    results_list = future.result()
                    if results_list:
                        all_geometry_results.extend(results_list)
                except Exception as exc:
                    print(f"\n! Ошибка в задаче для PDB {pdb_id}: {exc}")

        if not all_geometry_results:
            exit(
                "\n! Не удалось рассчитать геометрические параметры ни для одной структуры."
            )

        print(f"\nРассчитано {len(all_geometry_results)} взаимодействий (ароматика-ионон).")
        df_geometry = pd.DataFrame(all_geometry_results)  # Создаем DataFrame

        # --- 4. Сохранение результатов ---
        print(f"\n4. Сохранение таблицы с геометрией в {GEOMETRY_CSV_FILE}...")
        try:
            df_geometry.sort_values(
                by=["PDB_ID", "Ligand_Instance", "Neighbor_AA"], inplace=True
            )
            df_geometry.to_csv(
                GEOMETRY_CSV_FILE, index=False, encoding="utf-8", float_format="%.3f"
            )
            print("  Таблица успешно сохранена.")
        except Exception as e:
            print(f"! Ошибка сохранения CSV с геометрией: {e}")
    # Конец блока if df_geometry is None

    # --- 5. Визуализация результатов ---
    if df_geometry is not None and not df_geometry.empty:
        print("\n5. Визуализация результатов...")
        try:
            # Расстояние между центроидами
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_geometry, x="Centroid_Distance", kde=True, bins=30)
            plt.title(
                "Распределение расстояний между центроидами колец\n(Иононовое кольцо vs. Ароматическое АА)"
            )
            plt.xlabel("Расстояние (Ангстрем)")
            plt.ylabel("Количество взаимодействий")
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, "distance_centroid_distribution.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"  Гистограмма расстояний сохранена: {plot_path}")

            # Угол между плоскостями
            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=df_geometry,
                x="Interplane_Angle",
                kde=True,
                bins=30,
                binrange=(0, 90),
            )
            plt.title(
                "Распределение углов между плоскостями колец\n(Иононовое кольцо vs. Ароматическое АА)"
            )
            plt.xlabel(
                "Угол (градусы)\n0° - параллельно (стэкинг), 90° - перпендикулярно (T-shape)"
            )
            plt.ylabel("Количество взаимодействий")
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, "angle_interplane_distribution.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"  Гистограмма углов сохранена: {plot_path}")

            # Зависимость Угла от Расстояния (Scatter plot)
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=df_geometry,
                x="Centroid_Distance",
                y="Interplane_Angle",
                hue="AA_Type",
                alpha=0.6,
                s=50,
                palette="viridis",
            )
            plt.title(
                "Зависимость угла между плоскостями от расстояния между центроидами"
            )
            plt.xlabel("Расстояние между центроидами (Ангстрем)")
            plt.ylabel("Угол между плоскостями (градусы)")
            plt.ylim(0, 95)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(title="Ароматическая АА", loc="upper right")
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, "angle_vs_distance_scatter.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"  Scatter plot (угол vs расстояние) сохранен: {plot_path}")

            # --- ИЗМЕНЕНИЕ: Faceted гистограммы с НОРМИРОВКОЙ ---
            g = sns.displot(
                data=df_geometry,
                x="Interplane_Angle",
                col="AA_Type",
                kde=True,
                bins=20,
                binrange=(0, 90),
                stat="density",  # <--- Нормировка по плотности
                common_norm=False,  # <--- Нормировать каждую АА независимо
                col_wrap=3,
                height=4,
                aspect=1.2,
            )
            g.fig.suptitle("Нормированное распределение углов по типу АА", y=1.03)
            g.set_axis_labels("Угол (градусы)", "Плотность вероятности") # <--- Новая подпись Y
            g.set_titles("Аминокислота: {col_name}")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plot_path = os.path.join(PLOTS_DIR, "angle_distribution_density_by_aa.png") # Новое имя файла
            g.savefig(plot_path, dpi=300)
            plt.close()
            print(f"  Faceted НОРМИРОВАННАЯ гистограмма углов по АА сохранена: {plot_path}")
            # ----------------------------------------------------

        except Exception as e:
            print(f"! Ошибка при создании визуализаций: {e}")
            traceback.print_exc()
    else:
        print("\nНет данных геометрии для визуализации.")

    # --- Завершение ---
    main_end_time = time.time()
    print(
        f"\n--- Анализ геометрии завершен за {main_end_time - main_start_time:.2f} секунд ---"
    )