# -*- coding: utf-8 -*-
"""
Комбинированный скрипт v6 для анализа окружения каротиноидов:
- Учет фоновых частот аминокислот (Swiss-Prot).
- Расчет Log2 Обогащения (Observed/Expected).
- Статистические тесты (Хи-квадрат, Биномиальный тест с FDR).
- Отображение значимости на гистограммах.
- Heatmap и UMAP строятся на основе обогащения.
- Исправлена ошибка TypeError в binomtest (k должен быть int).
- Исправлена ошибка/предупреждение в chisquare (нормализация фоновых частот).
"""

import concurrent.futures
import os
import time
import traceback
import warnings
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# --- Новые импорты для статистики ---
try:
    import statsmodels.stats.multitest as smm
    from scipy.stats import binomtest, chisquare

    HAS_STATS = True
except ImportError:
    print(
        "! Предупреждение: SciPy или Statsmodels не найдены. Статистические тесты недоступны."
    )
    print("  Установите их: pip install scipy statsmodels")
    HAS_STATS = False
# ------------------------------------

# ... (остальные импорты как раньше) ...
try:
    import networkx as nx
except ImportError:
    exit("! Ошибка: NetworkX не найден. pip install networkx")
try:
    from Bio.PDB import Atom, MMCIFParser, NeighborSearch, Residue
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
except ImportError:
    exit("! Ошибка: Biopython не найден. pip install biopython")
try:
    import umap

    HAS_UMAP = True
except ImportError:
    print(
        "! Предупреждение: umap-learn не найден. UMAP недоступен. pip install umap-learn"
    )
    HAS_UMAP = False
try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist, squareform

    HAS_SCIPY = True
except ImportError:
    print(
        "! Предупреждение: SciPy не найден. Clustermap недоступен, расчет связей медленнее. pip install scipy"
    )
    HAS_SCIPY = False


# --- Конфигурация ---
CSV_FILE = "carotenoid_structures.csv"
CIF_DIR = "pdb_cif_files"
OUTPUT_BASE_DIR = "analysis_results_v6_enriched"  # Папка для результатов с обогащением
RING_DIR = os.path.join(OUTPUT_BASE_DIR, "ring_subst_env")
FULL_CHAIN_DIR = os.path.join(OUTPUT_BASE_DIR, "full_chain_env")
N_CORES = 16
DISTANCE_CUTOFF_NEIGHBORS = 4.5
BOND_DISTANCE_CUTOFF = 1.7
RING_SIZE = 6
ENRICHMENT_PLOT_NAME_ALPHA = "aa_enrichment_log2_alphabetical.png"
ENRICHMENT_PLOT_NAME_SCORE = "aa_enrichment_log2_by_score.png"
LIGAND_FACET_ENRICHMENT_NAME = "aa_enrichment_log2_by_ligand.png"
HEATMAP_ENRICHMENT_PLOT_NAME = "aa_clustermap_enrichment.png"
UMAP_ENRICHMENT_PLOT_NAME = "aa_umap_enrichment_ligand_colored.png"
RAW_DATA_NAME = "neighbor_residues_raw.csv"
# --- Новые имена файлов для статистики ---
STATS_OVERALL_NAME = "stats_overall_enrichment.csv"
STATS_PER_LIGAND_NAME = "stats_per_ligand_enrichment.csv"
# ------------------------------------------
PROCESS_LIMIT = None
warnings.simplefilter("ignore", PDBConstructionWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
np.seterr(
    divide="ignore", invalid="ignore"
)  # Оставляем, но log2(0) будет обрабатываться
STANDARD_AMINO_ACIDS = sorted(
    [
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
)
AA_INDEX_MAP = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}
ALPHA = 0.05  # Уровень значимости для стат. тестов

# ===============================================
# --- Фоновые Частоты Аминокислот (Swiss-Prot) ---
# ===============================================
# Из предоставленного изображения Table 1.
_SWISSPROT_AA_FREQUENCIES_RAW = {
    "ALA": 0.0777,
    "ARG": 0.0526,
    "ASN": 0.0427,
    "ASP": 0.0530,  # A R N D
    "CYS": 0.0157,
    "GLN": 0.0393,
    "GLU": 0.0656,
    "GLY": 0.0691,  # C Q E G
    "HIS": 0.0227,
    "ILE": 0.0591,
    "LEU": 0.0960,
    "LYS": 0.0595,  # H I L K
    "MET": 0.0238,
    "PHE": 0.0405,
    "PRO": 0.0469,
    "SER": 0.0694,  # M F P S
    "THR": 0.0550,
    "TRP": 0.0118,
    "TYR": 0.0311,
    "VAL": 0.0667,  # T W Y V
}
# --- Нормализация фоновых частот, чтобы сумма была ТОЧНО 1.0 ---
# Это важно для корректной работы Хи-квадрат теста
background_sum_raw = sum(_SWISSPROT_AA_FREQUENCIES_RAW.values())
SWISSPROT_AA_FREQUENCIES = {
    aa: freq / background_sum_raw for aa, freq in _SWISSPROT_AA_FREQUENCIES_RAW.items()
}
# ---------------------------------------------------------------

# Проверка
assert (
    len(SWISSPROT_AA_FREQUENCIES) == 20
), "Должно быть 20 аминокислот в фоновых частотах"
background_sum_norm = sum(SWISSPROT_AA_FREQUENCIES.values())
if abs(background_sum_norm - 1.0) > 1e-9:  # Используем более строгий допуск
    print(
        f"! Предупреждение: Сумма нормализованных фоновых частот ({background_sum_norm:.10f}) не равна 1.0."
    )


# ===============================================
# --- Функции Анализа (без изменений) ---
# ===============================================
def build_ligand_graph(residue: Residue, bond_cutoff: float = BOND_DISTANCE_CUTOFF):
    graph = nx.Graph()
    heavy_atoms = {a.serial_number: a for a in residue.get_atoms() if a.element != "H"}
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
                graph.add_edge(atom_list[i].serial_number, atom_list[j].serial_number)
    return graph, heavy_atoms


def find_ring_and_substituent_atoms(
    residue: Residue,
    ring_size: int = RING_SIZE,
    bond_cutoff: float = BOND_DISTANCE_CUTOFF,
):
    graph, heavy_atoms_dict = build_ligand_graph(residue, bond_cutoff)
    if not graph or not heavy_atoms_dict:
        return set()
    ring_atom_serials = set()
    all_target_atoms = set()
    try:
        cycles = list(nx.cycle_basis(graph))
        target_cycles = [c for c in cycles if len(c) == ring_size]
        if not target_cycles:
            cycles = list(nx.simple_cycles(graph))
            target_cycles = [c for c in cycles if len(c) == ring_size]
        for cycle in target_cycles:
            ring_atom_serials.update(cycle)
        if not ring_atom_serials:
            return set()
        substituent_atom_serials = set()
        for ring_sn in ring_atom_serials:
            if ring_sn not in graph:
                continue
            for neighbor_sn in graph.neighbors(ring_sn):
                if neighbor_sn not in ring_atom_serials:
                    substituent_atom_serials.add(neighbor_sn)
        all_target_serials = ring_atom_serials.union(substituent_atom_serials)
        all_target_atoms = {
            heavy_atoms_dict[sn] for sn in all_target_serials if sn in heavy_atoms_dict
        }
    except Exception:
        return set()
    return all_target_atoms


def process_structure(args):
    pdb_id, ligand_code, cif_path, neighbor_distance_cutoff, ring_size, bond_cutoff = (
        args
    )
    ring_neighbors_list = []
    full_chain_neighbors_list = []
    status = "Unknown Error"
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, cif_path)
        if not structure.child_list:
            return pdb_id, ligand_code, None, None, "No models"
        model = structure[0]
        ring_subst_atoms_search = set()
        all_ligand_heavy_atoms_search = set()
        protein_atoms = []
        ligand_residues_found = []
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith("H_") and residue.resname == ligand_code:
                    ligand_residues_found.append(residue)
                    current_target_atoms = find_ring_and_substituent_atoms(
                        residue, ring_size=ring_size, bond_cutoff=bond_cutoff
                    )
                    if current_target_atoms:
                        ring_subst_atoms_search.update(current_target_atoms)
                    for atom in residue.get_atoms():
                        if (
                            atom.element != "H"
                            and atom.coord is not None
                            and not np.isnan(atom.coord).any()
                        ):
                            all_ligand_heavy_atoms_search.add(atom)
                elif residue.id[0] == " " and residue.resname in AA_INDEX_MAP:
                    for atom in residue:
                        if atom.coord is not None and not np.isnan(atom.coord).any():
                            protein_atoms.append(atom)
        if not ligand_residues_found:
            return pdb_id, ligand_code, [], [], "Ligand not found"
        if not protein_atoms:
            return pdb_id, ligand_code, [], [], "No valid protein atoms"
        ns = None
        try:
            ns = NeighborSearch(protein_atoms)
        except Exception as e:
            return pdb_id, ligand_code, None, None, f"NeighborSearch error: {e}"
        if ns and ring_subst_atoms_search:
            n_atoms_ring = set()
            [
                n_atoms_ring.update(
                    ns.search(a.coord, neighbor_distance_cutoff, level="A")
                )
                for a in ring_subst_atoms_search
                if a.coord is not None
            ]
            n_res_ring = {
                a.get_parent()
                for a in n_atoms_ring
                if a.get_parent().id[0] == " "
                and a.get_parent().resname in AA_INDEX_MAP
            }
            for res in sorted(
                list(n_res_ring), key=lambda r: (r.get_parent().id, r.id[1], r.id[2])
            ):
                resn, ic = res.id[1], res.id[2].strip()
                r_id = f"{resn}{ic}" if ic else str(resn)
                ring_neighbors_list.append((res.get_parent().id, res.resname, r_id))
        if ns and all_ligand_heavy_atoms_search:
            n_atoms_full = set()
            [
                n_atoms_full.update(
                    ns.search(a.coord, neighbor_distance_cutoff, level="A")
                )
                for a in all_ligand_heavy_atoms_search
                if a.coord is not None
            ]
            n_res_full = {
                a.get_parent()
                for a in n_atoms_full
                if a.get_parent().id[0] == " "
                and a.get_parent().resname in AA_INDEX_MAP
            }
            for res in sorted(
                list(n_res_full), key=lambda r: (r.get_parent().id, r.id[1], r.id[2])
            ):
                resn, ic = res.id[1], res.id[2].strip()
                r_id = f"{resn}{ic}" if ic else str(resn)
                full_chain_neighbors_list.append(
                    (res.get_parent().id, res.resname, r_id)
                )
        status = "Success"
        if ligand_residues_found and not ring_subst_atoms_search:
            status = "Success (No rings/subst detected)"
        elif ligand_residues_found and not all_ligand_heavy_atoms_search:
            status = "Success (No valid ligand heavy atoms)"
        return (
            pdb_id,
            ligand_code,
            ring_neighbors_list,
            full_chain_neighbors_list,
            status,
        )
    except Exception as e:
        return (
            pdb_id,
            ligand_code,
            None,
            None,
            f"Unexpected error: {e} \n {traceback.format_exc()}",
        )


# ===============================================
# --- Новая Функция: Расчет Обогащения и Статистики ---
# ===============================================
def calculate_enrichment_and_stats(
    aa_counts_dict, total_neighbors, background_freqs, alpha=ALPHA
):
    """
    Рассчитывает Log2 обогащение, проводит Хи-квадрат и Биномиальные тесты.

    Args:
        aa_counts_dict (dict): Словарь {аминокислота: ЦЕЛОЕ ЧИСЛО}.
        total_neighbors (int): Общее число соседей (стандартных АА).
        background_freqs (dict): Словарь фоновых частот {аминокислота: частота}.
        alpha (float): Уровень значимости.

    Returns:
        tuple: (
            pd.Series: Log2 Обогащение (O/E),
            pd.Series: Скорректированные p-значения биномиального теста (FDR_BH),
            float: p-значение общего Хи-квадрат теста
        )
    """
    enrichment_scores = pd.Series(index=STANDARD_AMINO_ACIDS, dtype=float).fillna(
        -np.inf
    )  # Используем -inf для log2(0)
    p_values_binom = pd.Series(index=STANDARD_AMINO_ACIDS, dtype=float).fillna(1.0)
    observed_counts_list = []
    expected_counts_list = []
    valid_aa_for_chi2 = []

    if total_neighbors == 0 or not HAS_STATS:
        # Возвращаем нули/единицы, если нет данных или нет стат. библиотек
        enrichment_scores.replace(
            -np.inf, 0, inplace=True
        )  # Заменим -inf на 0 для консистентности
        return enrichment_scores, p_values_binom, 1.0

    # Биномиальный тест и расчет обогащения для каждой АА
    for aa in STANDARD_AMINO_ACIDS:
        # Убеждаемся, что obs_count - это integer
        obs_count = int(aa_counts_dict.get(aa, 0))
        bg_freq = background_freqs.get(aa, 0)

        observed_counts_list.append(obs_count)  # Собираем для Хи-квадрат

        if bg_freq > 0:
            exp_count = total_neighbors * bg_freq
            expected_counts_list.append(exp_count)  # Собираем для Хи-квадрат
            valid_aa_for_chi2.append(aa)  # Отмечаем АА для Хи-квадрат

            # Расчет обогащения
            if obs_count > 0:
                obs_freq = obs_count / total_neighbors
                enrichment_scores[aa] = np.log2(obs_freq / bg_freq)
            # else: оставляем -inf

            # Биномиальный тест (если возможно)
            if 0 < bg_freq < 1.0:  # p должно быть в (0, 1)
                try:
                    # Проверяем на обогащение (greater) или обеднение (less)
                    # Для двухстороннего: alternative='two-sided'
                    res_greater = binomtest(
                        obs_count, n=total_neighbors, p=bg_freq, alternative="greater"
                    )
                    res_less = binomtest(
                        obs_count, n=total_neighbors, p=bg_freq, alternative="less"
                    )
                    # Берем минимальное p-value, т.к. нас интересует любое значимое отклонение
                    p_values_binom[aa] = (
                        min(res_greater.pvalue, res_less.pvalue) * 2
                    )  # Умножаем на 2 для двухстороннего эквивалента
                    p_values_binom[aa] = min(
                        p_values_binom[aa], 1.0
                    )  # p-value не может быть > 1

                except ValueError as e:
                    # print(f"  Предупреждение: Binomtest для {aa} ({obs_count}/{total_neighbors}, p={bg_freq:.4f}) не удался: {e}")
                    p_values_binom[aa] = 1.0
            else:
                p_values_binom[aa] = 1.0  # Нельзя тестировать с p=0 или p=1
        else:
            expected_counts_list.append(0)  # Если фон. частота 0, ожидаемое 0
            # Обогащение не определено, если фон 0 (оставляем -inf)
            p_values_binom[aa] = 1.0  # Тест невозможен

    # Хи-квадрат тест на общее распределение
    obs_counts_chi2 = np.array(
        [int(aa_counts_dict.get(aa, 0)) for aa in valid_aa_for_chi2]
    )
    exp_counts_chi2 = np.array(
        [total_neighbors * background_freqs[aa] for aa in valid_aa_for_chi2]
    )

    chi2_p_value = 1.0
    if len(obs_counts_chi2) > 1 and np.all(exp_counts_chi2 > 0):
        # Проверка на низкие ожидаемые частоты
        if np.any(exp_counts_chi2 < 5):
            print(
                f"! Предупреждение: Хи-квадрат тест может быть неточным (ожидаемые частоты < 5). Сумма ожидаемых = {np.sum(exp_counts_chi2):.2f}"
            )
        try:
            # Проверка на совпадение сумм (из-за нормализации фона должно быть ОК)
            sum_obs = obs_counts_chi2.sum()
            sum_exp = exp_counts_chi2.sum()
            if not np.isclose(sum_obs, sum_exp):
                # print(f"Предупреждение: Суммы не совпадают для Chi2: Obs={sum_obs}, Exp={sum_exp:.4f}. Масштабирование ожидаемых.")
                # Масштабируем ожидаемые, чтобы сумма точно совпадала с наблюдаемой
                exp_counts_chi2 = exp_counts_chi2 * (sum_obs / sum_exp)

            chi2_stat, chi2_p_value = chisquare(
                f_obs=obs_counts_chi2, f_exp=exp_counts_chi2
            )

        except ValueError as e:
            # Ошибка все еще возможна, если суммы сильно расходятся по какой-то причине
            print(f"! Ошибка в Хи-квадрат расчете: {e}")
            chi2_p_value = 1.0

    elif len(obs_counts_chi2) <= 1:
        print("! Предупреждение: Недостаточно категорий (>1) для Хи-квадрат теста.")
    elif np.any(exp_counts_chi2 <= 0):
        print(
            "! Предупреждение: Нулевые или отрицательные ожидаемые частоты в Хи-квадрат тесте."
        )

    # Поправка на множественное тестирование (Benjamini/Hochberg FDR)
    valid_pvals = p_values_binom[
        p_values_binom <= 1.0
    ].values  # Берем только валидные p-values
    if len(valid_pvals) > 0:
        reject, pvals_corrected, _, _ = smm.multipletests(
            valid_pvals, alpha=alpha, method="fdr_bh"
        )
        # Обновляем Series скорректированными значениями
        corrected_p_values = pd.Series(1.0, index=STANDARD_AMINO_ACIDS)  # Иниц. 1.0
        valid_indices = p_values_binom[p_values_binom <= 1.0].index
        corrected_p_values.loc[valid_indices] = pvals_corrected
    else:
        corrected_p_values = p_values_binom  # Если нет валидных, оставляем 1.0

    # Заменяем -inf на разумное значение для графиков
    min_finite_score = enrichment_scores[np.isfinite(enrichment_scores)].min()
    if pd.isna(min_finite_score):
        min_finite_score = -5  # Запасной вариант, если все inf
    enrichment_scores.replace(
        -np.inf, min_finite_score - 1, inplace=True
    )  # Чуть ниже минимума

    return enrichment_scores, corrected_p_values, chi2_p_value


# ===============================================
# --- Функции Визуализации (МОДИФИЦИРОВАНЫ) ---
# ===============================================


def plot_enrichment_scores(
    aa_counts,
    total_neighbors,
    filename="aa_enrichment.png",
    title_suffix="",
    sort_order="alphabetical",
):
    """Строит гистограмму Log2 Обогащения (O/E) со статистикой."""
    if not aa_counts or total_neighbors == 0:
        print(f"Нет данных для гистограммы обогащения {title_suffix}.")
        return None
    if not HAS_STATS:
        print(f"Статистика пропущена (нет SciPy/Statsmodels) для {title_suffix}")
        return None

    # --- Расчет статистики и обогащения ---
    enrichment_scores, p_values_corrected, chi2_p_value = (
        calculate_enrichment_and_stats(
            aa_counts, total_neighbors, SWISSPROT_AA_FREQUENCIES
        )
    )
    stats_df = pd.DataFrame(
        {
            "AA": enrichment_scores.index,
            "Log2_Enrichment(O/E)": enrichment_scores.values,
            "p_value_corrected(FDR_BH)": p_values_corrected.values,
        }
    ).sort_values(by="AA")
    # -------------------------------------

    plot_data = enrichment_scores.copy()
    plot_order = None
    title_prefix = "Log2 Обогащение"
    if sort_order == "alphabetical":
        plot_order = STANDARD_AMINO_ACIDS
        title_prefix += " (алфавит)"
    elif sort_order == "score":
        plot_data.sort_values(ascending=False, inplace=True)
        plot_order = plot_data.index.tolist()
        title_prefix += " (по убыванию)"
    else:
        plot_order = STANDARD_AMINO_ACIDS
        title_prefix += " (алфавит)"

    plt.figure(figsize=(14, 9))  # Чуть выше для заголовка и звезд
    ax = sns.barplot(
        x=plot_data.index, y=plot_data.values, order=plot_order, palette="coolwarm"
    )  # Палитра для +/-
    plt.axhline(0, color="grey", lw=1.0, linestyle="--")  # Линия нуля
    plt.xlabel("Аминокислота", fontsize=14)
    plt.ylabel(f"Log2 Обогащение (Набл./Ожид.) {title_suffix}", fontsize=14)
    overall_stat_sig = chi2_p_value < ALPHA
    chi2_sig_str = f"p={chi2_p_value:.3g}{' (*)' if overall_stat_sig else ''}"
    plt.title(
        f"{title_prefix} аминокислот в окружении ({DISTANCE_CUTOFF_NEIGHBORS} Å) {title_suffix}\nОбщий Хи-квадрат: {chi2_sig_str}",
        fontsize=16,
    )
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    # --- Добавление звездочек значимости ---
    y_lim = ax.get_ylim()
    y_range = y_lim[1] - y_lim[0]
    offset = y_range * 0.02  # Небольшой отступ для звезд

    for i, bar in enumerate(ax.patches):
        aa = plot_order[i]
        p_val = p_values_corrected.get(aa, 1.0)
        score = plot_data.get(aa, 0)
        stars = ""
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < ALPHA:
            stars = "*"

        if stars:
            height = bar.get_height()
            # Размещаем звезды над баром (если >0) или под баром (если <0)
            text_y = height + offset if height >= 0 else height - offset
            va = "bottom" if height >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                text_y,
                stars,
                ha="center",
                va=va,
                color="black",
                fontsize=10,
                weight="bold",
            )
    # ---------------------------------------

    # Установим симметричные пределы Y, если есть и положительные и отрицательные значения
    max_abs_score = plot_data.abs().max()
    if plot_data.min() < 0 and plot_data.max() > 0 and max_abs_score > 0:
        ax.set_ylim(-max_abs_score * 1.1, max_abs_score * 1.1)
    else:
        ax.set_ylim(y_lim)  # Восстанавливаем исходные, если только +/- или все 0

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Оставляем место сверху для заголовка
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        print(f"Гистограмма обогащения ({sort_order}) сохранена: {filename}")
        plt.close()
        return stats_df  # Возвращаем DataFrame со статистикой
    except Exception as e:
        print(
            f"! Ошибка сохранения гистограммы обогащения ({sort_order}) {filename}: {e}"
        )
        return None


def plot_ligand_faceted_histograms(
    df_neighbors, filename="aa_enrichment_by_ligand.png", title_suffix=""
):
    """Строит faceted-график с гистограммами ОБОГАЩЕНИЯ для каждого лиганда."""
    if df_neighbors is None or df_neighbors.empty:
        print(f"Нет данных для faceted-гистограмм {title_suffix}.")
        return None
    if not HAS_STATS:
        print(
            f"Статистика пропущена (нет SciPy/Statsmodels) для faceted {title_suffix}"
        )
        return None

    ligand_aa_counts = (
        df_neighbors.groupby(["ligand", "resname"]).size().unstack(fill_value=0)
    )
    # Убедимся, что все стандартные аминокислоты есть как колонки
    ligand_aa_counts = ligand_aa_counts.reindex(
        columns=STANDARD_AMINO_ACIDS, fill_value=0
    )
    ligand_totals = ligand_aa_counts.sum(axis=1)

    # Фильтруем лиганды без соседей
    valid_ligands = ligand_totals[ligand_totals > 0].index
    if len(valid_ligands) == 0:
        print(f"Нет лигандов с соседями для faceted-гистограмм {title_suffix}.")
        return None

    print(f"Расчет обогащения и статистики для {len(valid_ligands)} лигандов...")
    ligand_enrichment_data = {}
    ligand_p_values = {}
    ligand_chi2_p = {}
    all_ligand_stats = []

    for ligand in tqdm(valid_ligands, desc="Статистика по лигандам", leave=False):
        counts_series = ligand_aa_counts.loc[ligand]
        total = ligand_totals[ligand]
        scores, pvals, chi2p = calculate_enrichment_and_stats(
            counts_series.to_dict(), total, SWISSPROT_AA_FREQUENCIES
        )
        ligand_enrichment_data[ligand] = scores
        ligand_p_values[ligand] = pvals
        ligand_chi2_p[ligand] = chi2p
        # Собираем статистику для таблицы
        stats_df_lig = pd.DataFrame(
            {
                "AA": scores.index,
                "Log2_Enrichment(O/E)": scores.values,
                "p_value_corrected(FDR_BH)": pvals.values,
            }
        )
        stats_df_lig["Ligand"] = ligand
        stats_df_lig["Total_Neighbors"] = total
        stats_df_lig["Chi2_p_value"] = chi2p
        all_ligand_stats.append(stats_df_lig)

    if not all_ligand_stats:
        print("Не удалось рассчитать статистику ни для одного лиганда.")
        return None
    combined_stats_df = pd.concat(all_ligand_stats, ignore_index=True)

    # Подготовка данных для catplot
    enrichment_df_meltable = pd.DataFrame(ligand_enrichment_data).T  # Лиганды в строках
    enrichment_df_meltable = enrichment_df_meltable.reset_index().rename(
        columns={"index": "ligand"}
    )
    plot_data_enrichment = enrichment_df_meltable.melt(
        id_vars="ligand", var_name="Amino Acid", value_name="Log2 Enrichment (O/E)"
    )

    n_ligands = len(valid_ligands)
    col_wrap = min(n_ligands, 5 if n_ligands > 4 else n_ligands)
    plot_height = 4
    aspect_ratio = 1.6

    print(f"Построение faceted-гистограммы обогащения для {n_ligands} лигандов...")
    try:
        g = sns.catplot(
            data=plot_data_enrichment,
            x="Amino Acid",
            y="Log2 Enrichment (O/E)",
            col="ligand",
            kind="bar",
            col_wrap=col_wrap,
            height=plot_height,
            aspect=aspect_ratio,
            palette="coolwarm",  # Используем coolwarm
            sharey=True,  # Важно: делаем оси Y одинаковыми для сравнения обогащения
            order=STANDARD_AMINO_ACIDS,
            sharex=True,
        )

        g.set_axis_labels("Аминокислота", "Log2 Обогащение (Набл./Ожид.)", fontsize=12)
        # Добавляем общую стат. значимость лиганда в заголовок facet'а
        for ax, ligand_name in zip(g.axes.flat, g.col_names):
            if (
                ligand_name not in ligand_chi2_p
            ):  # Проверка на случай, если лиганд был отфильтрован
                continue
            chi2p = ligand_chi2_p.get(ligand_name, 1.0)
            chi2_sig_str = f" (Chi2 p={chi2p:.2g}{'*' if chi2p < ALPHA else ''})"
            ax.set_title(f"Лиганд: {ligand_name}{chi2_sig_str}", size=13)

            # Добавляем звездочки на бары внутри каждого facet'а
            pvals_ligand = ligand_p_values.get(ligand_name)
            if pvals_ligand is not None:
                current_bars = [
                    p for p in ax.patches if p.get_height() != 0
                ]  # Только видимые бары
                y_lim_facet = ax.get_ylim()
                y_range_facet = y_lim_facet[1] - y_lim_facet[0]
                offset_facet = y_range_facet * 0.02

                # Проверяем, что число баров соответствует числу АА
                if len(current_bars) == len(STANDARD_AMINO_ACIDS):
                    for i, bar in enumerate(current_bars):
                        aa = STANDARD_AMINO_ACIDS[i]  # Порядок соответствует order
                        p_val = pvals_ligand.get(aa, 1.0)
                        score = bar.get_height()  # Получаем высоту прямо из бара
                        stars = ""
                        if p_val < 0.001:
                            stars = "***"
                        elif p_val < 0.01:
                            stars = "**"
                        elif p_val < ALPHA:
                            stars = "*"

                        if stars:
                            text_y = (
                                score + offset_facet
                                if score >= 0
                                else score - offset_facet
                            )
                            va = "bottom" if score >= 0 else "top"
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                text_y,
                                stars,
                                ha="center",
                                va=va,
                                color="black",
                                fontsize=8,
                                weight="bold",
                            )  # Меньше шрифт

            # Явно устанавливаем метки X и их видимость
            ax.axhline(0, color="grey", lw=0.8, linestyle="--")  # Линия нуля
            ax.set_xticks(range(len(STANDARD_AMINO_ACIDS)))
            ax.set_xticklabels(
                STANDARD_AMINO_ACIDS, rotation=70, ha="right", fontsize=18
            )  # Меньше шрифт
            plt.setp(ax.get_xticklabels(), visible=True)
            ax.tick_params(axis="y", labelsize=10)

        plt.suptitle(
            f"Log2 Обогащение АА по лигандам {title_suffix}", y=1.03, fontsize=16
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        g.savefig(filename, dpi=300)
        print(f"Faceted-гистограмма обогащения по лигандам сохранена: {filename}")
        plt.close(g.fig)
        return combined_stats_df  # Возвращаем статистику по лигандам
    except Exception as e:
        print(f"\n! Ошибка построения faceted-гистограммы обогащения {filename}: {e}")
        traceback.print_exc()
        return None


def plot_clustered_heatmap(
    data_matrix,
    index_labels,
    filename="aa_clustermap_enrichment.png",
    title_suffix="",
):
    """Строит кластеризованную тепловую карту ОБОГАЩЕНИЯ."""
    if not HAS_SCIPY:
        print(f"Clustermap {title_suffix} пропущен (требуется SciPy).")
        return
    # Матрица уже должна быть матрицей обогащения
    # Используем стандартное отклонение по строке > 0 как критерий валидности
    valid_rows_mask = np.nanstd(data_matrix, axis=1) > 1e-9
    n_original, n_filtered = data_matrix.shape[0], valid_rows_mask.sum()
    if n_filtered < n_original:
        print(
            f"Clustermap {title_suffix}: Отфильтровано {n_original - n_filtered} структур (нет вариации в обогащении)."
        )
    if n_filtered < 2 or data_matrix.shape[1] < 2:
        print(
            f"Недостаточно данных для clustermap обогащения {title_suffix} ({n_filtered}x{data_matrix.shape[1]})."
        )
        return

    filtered_matrix = data_matrix[valid_rows_mask, :]
    filtered_labels = [
        label for i, label in enumerate(index_labels) if valid_rows_mask[i]
    ]

    # Заменяем NaN на 0 для визуализации (или на другое значение)
    filtered_matrix_plot = np.nan_to_num(filtered_matrix, nan=0.0)

    try:
        row_height_factor = 0.25
        fig_height = max(15, n_filtered * row_height_factor)
        fig_width = 20
        # Используем метрику, устойчивую к выбросам
        g = sns.clustermap(
            filtered_matrix_plot,
            cmap="coolwarm",
            center=0,  # Центрируем палитру на 0
            linewidths=0.5,
            metric="correlation",
            method="average",
            figsize=(fig_width, fig_height),
            xticklabels=STANDARD_AMINO_ACIDS,
            yticklabels=filtered_labels,
            dendrogram_ratio=(0.08, 0.12),
            # z_score=0, # Опционально: стандартизация по строкам
        )

        base_fontsize = 48  # Уменьшаем базовый шрифт
        xlabel_fontsize = base_fontsize
        ylabel_fontsize = base_fontsize
        xtick_fontsize = base_fontsize - 2
        ytick_fontsize = max(8, base_fontsize - n_filtered // 20)  # Адаптируем Y

        g.ax_heatmap.set_xlabel("Аминокислота в окружении", fontsize=xlabel_fontsize)
        g.ax_heatmap.set_ylabel("Структура (PDB ID)", fontsize=ylabel_fontsize)
        plt.suptitle(
            f"Кластеризованная тепловая карта Log2 Обогащения АА {title_suffix}",
            y=1.01,
            fontsize=base_fontsize + 2,
        )

        plt.setp(
            g.ax_heatmap.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=xtick_fontsize,
        )
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=ytick_fontsize)
        g.cax.set_title(
            "Log2(O/E)", fontsize=base_fontsize - 2
        )  # Подпись к цветовой шкале

        g.fig.tight_layout(rect=[0, 0.02, 1, 0.98])

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        g.savefig(filename, dpi=300)
        print(f"Clustermap обогащения сохранен: {filename}")
        plt.close(g.fig)
    except Exception as e:
        print(f"\n! Ошибка построения clustermap обогащения {filename}: {e}")
        traceback.print_exc()


def plot_umap_results(
    embedding,
    labels,
    color_map_title,
    filename="umap_enrichment_plot.png",
    title="UMAP проекция профилей обогащения",
):
    """Строит UMAP-график (данные основаны на обогащении)."""
    # ... (Эта функция остается почти без изменений) ...
    if embedding is None or embedding.shape[0] == 0:
        print(f"Нет данных для UMAP ({os.path.basename(filename)}).")
        return
    plt.figure(figsize=(12, 10))
    unique_labels = sorted(list(set(labels)))
    if not unique_labels:
        print(f"Нет меток для UMAP ({os.path.basename(filename)}).")
        plt.close()
        return
    palette = sns.color_palette("husl", len(unique_labels))
    lut = dict(zip(unique_labels, palette))
    colors = [lut.get(l, "#000000") for l in labels]
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=25,
        alpha=0.8,
        edgecolors="w",
        linewidth=0.5,
    )
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
        handles=handles,
        title=color_map_title,
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)  # Заголовок изменен по умолчанию
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        print(f"UMAP график (обогащение) сохранен: {filename}")
        plt.close()
    except Exception as e:
        print(f"\n! Ошибка сохранения UMAP (обогащение) {filename}: {e}")


# ===============================================
# --- Функции Сохранения/Загрузки Таблиц ---
# ===============================================
def save_results_to_csv(results_dict, ligand_map, output_dir, filename):
    """Сохраняет детальные результаты (сырые соседи) в CSV."""
    output_path = os.path.join(output_dir, filename)
    all_neighbors_data = []
    for pdb_id, neighbors in results_dict.items():
        if not neighbors:
            continue
        ligand = ligand_map.get(pdb_id, "Unknown")
        for chain, resname, resnum in neighbors:
            all_neighbors_data.append(
                {
                    "pdbID": pdb_id,
                    "Ligand": ligand,
                    "Neighbor_Chain": chain,
                    "Neighbor_Resname": resname,
                    "Neighbor_Resnum": resnum,
                }
            )
    if not all_neighbors_data:
        print(f"Нет данных о соседях для {output_path}.")
        return False
    try:
        df_results = pd.DataFrame(all_neighbors_data)
        df_results.sort_values(
            by=["pdbID", "Neighbor_Chain", "Neighbor_Resnum", "Neighbor_Resname"],
            inplace=True,
        )
        os.makedirs(output_dir, exist_ok=True)
        df_results.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Таблица ({len(df_results)} записей) сохранена: {output_path}")
        return True
    except Exception as e:
        print(f"\n! Ошибка сохранения CSV {output_path}: {e}")
        return False


def load_results_from_csv(output_dir, filename, pdb_ids_needed):
    """Загружает результаты (сырые соседи) из CSV и форматирует в словарь."""
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        return None
    print(f"Загрузка кэшированных данных из: {input_path}")
    try:
        df_results = pd.read_csv(
            input_path, dtype={"pdbID": str, "Neighbor_Resnum": str}
        )
        # Убедимся, что Neighbor_Resnum прочитался как строка
        df_results["Neighbor_Resnum"] = df_results["Neighbor_Resnum"].astype(str)

        df_results = df_results[df_results["pdbID"].isin(pdb_ids_needed)]
        results_dict = defaultdict(list)
        # Используем итерацию вместо groupby
        for index, row in tqdm(
            df_results.iterrows(),
            total=len(df_results),
            desc=f"Форматирование кэша {os.path.basename(input_path)}",
            leave=False,
        ):
            # Проверяем наличие обязательных колонок
            if not all(
                col in row
                for col in [
                    "pdbID",
                    "Neighbor_Chain",
                    "Neighbor_Resname",
                    "Neighbor_Resnum",
                ]
            ):
                print(
                    f"! Предупреждение: Пропущена строка в кэше из-за отсутствия колонок: {row}"
                )
                continue
            results_dict[row["pdbID"]].append(
                (row["Neighbor_Chain"], row["Neighbor_Resname"], row["Neighbor_Resnum"])
            )

        # Добавляем пустые списки для PDB без соседей, которые были запрошены
        for pdb_id in pdb_ids_needed:
            if pdb_id not in results_dict:
                results_dict[pdb_id] = []
        return dict(results_dict)
    except Exception as e:
        print(
            f"\n! Ошибка загрузки/форматирования кэша {input_path}: {e}. Будет пересчет."
        )
        return None


# ===============================================
# --- Основной блок выполнения ---
# ===============================================
if __name__ == "__main__":
    main_start_time = time.time()
    print(f"--- Анализ окружения каротиноидов v6 (Обогащение и Статистика) ---")
    # Используем новую папку для результатов
    print(f"Базовая директория результатов: {OUTPUT_BASE_DIR}")
    os.makedirs(RING_DIR, exist_ok=True)
    os.makedirs(FULL_CHAIN_DIR, exist_ok=True)

    print(f"\nЧтение метаданных из: {CSV_FILE}")
    # ... (Загрузка CSV и PDB ID как раньше) ...
    if not os.path.exists(CSV_FILE):
        exit(f"! Ошибка: Файл {CSV_FILE} не найден!")
    if not os.path.exists(CIF_DIR):
        exit(f"! Ошибка: Папка {CIF_DIR} не найдена!")
    try:
        df = pd.read_csv(CSV_FILE, dtype={"pdbID": str})
        df["pdbID"] = df["pdbID"].str.strip()
        print(f"Загружено {len(df)} записей.")
        pdb_ligand_map = (
            df.drop_duplicates(subset=["pdbID"]).set_index("pdbID")["Ligand"].to_dict()
        )
    except Exception as e:
        exit(f"! Ошибка чтения CSV файла: {e}")

    pdb_ids_to_process = sorted(df["pdbID"].unique())
    if PROCESS_LIMIT:
        pdb_ids_to_process = pdb_ids_to_process[:PROCESS_LIMIT]
        print(f"Ограничение: первые {PROCESS_LIMIT} PDB ID.")

    # --- Проверка КЭШа и Запуск Обработки ---
    # Кэш сырых данных ищем в новой папке (или старой, если ее указать)
    cache_base_dir = OUTPUT_BASE_DIR  # Ищем кэш там же, куда сохраняем
    cache_ring_dir = os.path.join(cache_base_dir, "ring_subst_env")
    cache_full_chain_dir = os.path.join(cache_base_dir, "full_chain_env")
    ring_cache_file = os.path.join(cache_ring_dir, RAW_DATA_NAME)
    full_chain_cache_file = os.path.join(cache_full_chain_dir, RAW_DATA_NAME)

    ring_results = None
    full_chain_results = None
    load_from_cache = False
    if os.path.exists(ring_cache_file) and os.path.exists(full_chain_cache_file):
        print(
            f"\nНайдены кэш-файлы в '{cache_base_dir}'. Загрузка сырых данных о соседях..."
        )
        ring_results = load_results_from_csv(
            cache_ring_dir, RAW_DATA_NAME, set(pdb_ids_to_process)
        )
        full_chain_results = load_results_from_csv(
            cache_full_chain_dir, RAW_DATA_NAME, set(pdb_ids_to_process)
        )
        if (
            ring_results is not None
            and full_chain_results is not None
            and len(ring_results) == len(pdb_ids_to_process)
            and len(full_chain_results) == len(pdb_ids_to_process)
        ):
            print("Сырые данные о соседях успешно загружены из кэша.")
            load_from_cache = True
        else:
            print(
                "Ошибка загрузки кэша сырых данных или не все PDB ID найдены, будет пересчет."
            )

    if not load_from_cache:
        # ... (Параллельная обработка как раньше) ...
        print(
            "\nКэш не найден/неполный. Запуск полного анализа для получения сырых данных..."
        )
        tasks = []
        skipped_files = 0
        print(f"Подготовка задач для {len(pdb_ids_to_process)} PDB ID...")
        for pdb_id in pdb_ids_to_process:
            ligand_code = pdb_ligand_map.get(pdb_id)
            if not ligand_code:
                continue
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
                skipped_files += 1
        if skipped_files > 0:
            print(
                f"! Предупреждение: Пропущено {skipped_files} PDB ID из-за отсутствия CIF."
            )
        if not tasks:
            exit("! Нет задач для обработки.")
        print(
            f"\nЗапуск параллельной обработки ({len(tasks)} задач) на {N_CORES} ядрах..."
        )
        ring_results = {}
        full_chain_results = {}
        success_count = 0
        fail_count = 0
        status_counts = Counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_CORES) as executor:
            results_iterator = executor.map(process_structure, tasks)
            for result in tqdm(
                results_iterator, total=len(tasks), desc="Обработка PDB"
            ):
                try:
                    (
                        pdb_id_res,
                        _,
                        ring_neighbors,
                        full_neighbors,
                        status,
                    ) = result
                    status_counts[status] += 1

                    # Обработка случая, когда neighbors = None
                    error_occurred = (
                        status.startswith("Unexpected error")
                        or status == "Unknown Error"
                        or "error" in status.lower()
                    )

                    if ring_neighbors is not None:
                        ring_results[pdb_id_res] = ring_neighbors
                    else:
                        ring_results[pdb_id_res] = []  # Пустой список при None
                        if (
                            error_occurred and fail_count == success_count
                        ):  # Считаем ошибку один раз
                            fail_count += 1

                    if full_neighbors is not None:
                        full_chain_results[pdb_id_res] = full_neighbors
                    else:
                        full_chain_results[pdb_id_res] = []
                        if (
                            error_occurred and fail_count == success_count
                        ):  # Считаем ошибку один раз
                            fail_count += 1

                    if not error_occurred:
                        success_count += 1
                    # Если была ошибка, fail_count уже увеличен выше

                except Exception as e_main:
                    print(f"\n! Критическая ошибка при обработке результата: {e_main}")
                    traceback.print_exc()
                    fail_count += 1  # Считаем как провал
                    status_counts["Result Handling Error"] += 1

        print("\n--- Статистика Параллельной Обработки ---")
        # Корректируем счетчики, чтобы сумма совпадала с total
        actual_processed = success_count + fail_count
        if actual_processed != len(tasks):
            print(
                f"! Предупреждение: число обработанных ({actual_processed}) не совпадает с числом задач ({len(tasks)})"
            )

        print(f"Успешно обработано PDB ID (статус не ошибка): {success_count}")
        print(f"Ошибки обработки (явные ошибки или None результат): {fail_count}")
        print("Статусы обработки:")
        [print(f"  - {st}: {cnt}") for st, cnt in sorted(status_counts.items())]

        # Сохраняем сырые данные в НОВУЮ папку
        print(f"Сохранение кэша сырых данных в: {OUTPUT_BASE_DIR}")
        if ring_results:
            save_results_to_csv(ring_results, pdb_ligand_map, RING_DIR, RAW_DATA_NAME)
        if full_chain_results:
            save_results_to_csv(
                full_chain_results, pdb_ligand_map, FULL_CHAIN_DIR, RAW_DATA_NAME
            )

    # --- Пост-обработка и Визуализация (с расчетом обогащения) ---
    if not ring_results and not full_chain_results:
        exit("\n! Нет данных для анализа.")
    analysis_types = [
        ("Кольца+Заместители", ring_results, RING_DIR),
        ("Вся Цепь", full_chain_results, FULL_CHAIN_DIR),
    ]

    for analysis_name, current_results, output_dir in analysis_types:
        print(f"\n{'='*15} Анализ Обогащения и Визуализация: {analysis_name} {'='*15}")
        print(f"--- Результаты сохраняются в: {output_dir} ---")
        if not current_results:
            print("Нет данных для этого типа анализа.")
            continue

        all_neighbors_raw_flat = []
        structure_aa_counts = {}
        structure_labels = {}
        valid_pdb_ids_for_analysis = sorted(current_results.keys())
        print("Агрегация данных и создание векторов СЧЕТЧИКОВ...")
        for pdb_id in tqdm(valid_pdb_ids_for_analysis, desc="Агрегация", leave=False):
            neighbors = current_results[pdb_id]
            ligand = pdb_ligand_map.get(pdb_id, "Unknown")
            structure_labels[pdb_id] = ligand
            # Собираем данные для DataFrame (используется для faceted plot)
            for _, resname, _ in neighbors:
                if resname in AA_INDEX_MAP:  # Только стандартные АА
                    all_neighbors_raw_flat.append(
                        {"resname": resname, "pdbID": pdb_id, "ligand": ligand}
                    )

            # Считаем СЧЕТЧИКИ (ЦЕЛЫЕ ЧИСЛА) для каждой структуры
            aa_vector = np.zeros(
                len(STANDARD_AMINO_ACIDS), dtype=int
            )  # <--- ВАЖНО: dtype=int
            if neighbors:
                counts = Counter(
                    resname for _, resname, _ in neighbors if resname in AA_INDEX_MAP
                )
                for aa, index in AA_INDEX_MAP.items():
                    aa_vector[index] = counts.get(aa, 0)
            structure_aa_counts[pdb_id] = aa_vector  # Храним целые счетчики

        if not all_neighbors_raw_flat:
            print("Нет стандартных аминокислот в окружении для анализа.")
            continue
        df_all_neighbors = pd.DataFrame(all_neighbors_raw_flat)
        print(
            f"Всего стандартных АА остатков в окружении ({analysis_name}): {len(df_all_neighbors)}"
        )

        # --- Общие Гистограммы Обогащения и Статистика ---
        overall_aa_counts_counter = Counter(df_all_neighbors["resname"])
        overall_aa_counts = {
            aa: overall_aa_counts_counter.get(aa, 0) for aa in STANDARD_AMINO_ACIDS
        }  # Словарь с int
        total_overall_neighbors = sum(overall_aa_counts.values())
        print("\n--- Общая статистика обогащения ---")
        overall_stats_df = plot_enrichment_scores(
            overall_aa_counts,
            total_overall_neighbors,
            filename=os.path.join(output_dir, ENRICHMENT_PLOT_NAME_ALPHA),
            title_suffix=f"({analysis_name} - все)",
            sort_order="alphabetical",
        )
        _ = plot_enrichment_scores(  # Повторный вызов для другой сортировки
            overall_aa_counts,
            total_overall_neighbors,
            filename=os.path.join(output_dir, ENRICHMENT_PLOT_NAME_SCORE),
            title_suffix=f"({analysis_name} - все)",
            sort_order="score",
        )
        if overall_stats_df is not None:
            stats_filename = os.path.join(output_dir, STATS_OVERALL_NAME)
            try:
                overall_stats_df.to_csv(
                    stats_filename, index=False, encoding="utf-8", float_format="%.6g"
                )
                print(f"Таблица общей статистики сохранена: {stats_filename}")
            except Exception as e:
                print(
                    f"! Ошибка сохранения таблицы общей статистики {stats_filename}: {e}"
                )

        # --- Faceted Гистограммы Обогащения по Лигандам ---
        print("\n--- Статистика обогащения по лигандам ---")
        per_ligand_stats_df = plot_ligand_faceted_histograms(
            df_all_neighbors,
            filename=os.path.join(output_dir, LIGAND_FACET_ENRICHMENT_NAME),
            title_suffix=f"({analysis_name})",
        )
        if per_ligand_stats_df is not None:
            stats_filename_lig = os.path.join(output_dir, STATS_PER_LIGAND_NAME)
            try:
                # Добавим колонку тип анализа
                per_ligand_stats_df["Analysis_Type"] = analysis_name
                # Сортировка для удобства просмотра
                per_ligand_stats_df = per_ligand_stats_df.sort_values(
                    by=["Ligand", "AA"]
                )
                per_ligand_stats_df.to_csv(
                    stats_filename_lig,
                    index=False,
                    encoding="utf-8",
                    float_format="%.6g",
                )
                print(f"Таблица статистики по лигандам сохранена: {stats_filename_lig}")
            except Exception as e:
                print(
                    f"! Ошибка сохранения таблицы статистики по лигандам {stats_filename_lig}: {e}"
                )

        # --- Подготовка Матрицы ОБОГАЩЕНИЯ для Heatmap и UMAP ---
        if not valid_pdb_ids_for_analysis:
            print("Нет PDB ID для Heatmap/UMAP.")
            continue
        print("\nРасчет матрицы обогащения для структур...")
        enrichment_matrix_list = []
        labels_list = []  # Метки для строк матрицы
        valid_pdb_ids_for_matrix = []  # ID для строк матрицы

        for pdb_id in tqdm(
            valid_pdb_ids_for_analysis, desc="Матрица обогащения", leave=False
        ):
            aa_counts_vector = structure_aa_counts.get(pdb_id)
            if aa_counts_vector is None:
                continue  # Пропускаем, если нет данных

            total_neighbors_struct = int(aa_counts_vector.sum())
            # Создаем словарь с int значениями
            aa_counts_struct_dict = {
                aa: int(count)
                for aa, count in zip(STANDARD_AMINO_ACIDS, aa_counts_vector)
            }

            # Рассчитываем обогащение для данной структуры
            scores, _, _ = calculate_enrichment_and_stats(
                aa_counts_struct_dict, total_neighbors_struct, SWISSPROT_AA_FREQUENCIES
            )
            enrichment_matrix_list.append(scores.values)  # Добавляем массив Log2(O/E)
            labels_list.append(structure_labels[pdb_id])
            valid_pdb_ids_for_matrix.append(pdb_id)

        if not enrichment_matrix_list:
            print("Не удалось создать матрицу обогащения.")
            continue
        enrichment_matrix = np.array(
            enrichment_matrix_list
        )  # Финальная матрица (структуры x аминокислоты)

        # --- Clustermap на основе ОБОГАЩЕНИЯ ---
        plot_clustered_heatmap(
            enrichment_matrix,
            index_labels=valid_pdb_ids_for_matrix,
            filename=os.path.join(output_dir, HEATMAP_ENRICHMENT_PLOT_NAME),
            title_suffix=f"({analysis_name})",
        )

        # --- UMAP на основе ОБОГАЩЕНИЯ ---
        if HAS_UMAP:
            # Используем ту же маску валидности, что и для heatmap
            valid_rows_mask_umap = np.nanstd(enrichment_matrix, axis=1) > 1e-9
            n_orig, n_filt = enrichment_matrix.shape[0], valid_rows_mask_umap.sum()
            if n_filt < n_orig:
                print(
                    f"UMAP {analysis_name}: Отфильтровано {n_orig - n_filt} структур (нет вариации в обогащении)."
                )

            if n_filt > 1 and enrichment_matrix.shape[1] > 1:
                # Используем данные обогащения, заменяя NaN на 0 для UMAP
                umap_data = np.nan_to_num(
                    enrichment_matrix[valid_rows_mask_umap, :], nan=0.0
                )
                umap_labels = [
                    l for i, l in enumerate(labels_list) if valid_rows_mask_umap[i]
                ]
                print(
                    f"Запуск UMAP для {analysis_name} ({n_filt} точек) на данных обогащения..."
                )
                try:
                    # Метрика 'cosine' хорошо работает для векторов профилей
                    reducer = umap.UMAP(
                        n_neighbors=min(15, n_filt - 1) if n_filt > 1 else 1,
                        min_dist=0.1,
                        n_components=2,
                        metric="cosine",  # Пробуем cosine
                        random_state=42,
                        low_memory=True,
                    )
                    embedding = reducer.fit_transform(umap_data)
                    plot_umap_results(
                        embedding,
                        labels=umap_labels,
                        color_map_title="Ligand",
                        filename=os.path.join(output_dir, UMAP_ENRICHMENT_PLOT_NAME),
                        title=f"UMAP проекция профилей обогащения ({analysis_name})",
                    )
                except Exception as e:
                    print(f"! Ошибка UMAP (обогащение) для {analysis_name}: {e}")
            else:
                print(
                    f"UMAP для {analysis_name} (обогащение) пропущен (мало данных после фильтрации: {n_filt}x{enrichment_matrix.shape[1]})."
                )
        else:
            print(f"UMAP для {analysis_name} (обогащение) пропущен (нет umap-learn).")

        # --- Сохранение таблицы сырых данных (перезапись, только если не из кэша) ---
        raw_data_path = os.path.join(output_dir, RAW_DATA_NAME)
        if not load_from_cache or not os.path.exists(raw_data_path):
            print(f"Сохранение/перезапись сырых данных о соседях в: {raw_data_path}")
            save_results_to_csv(
                current_results, pdb_ligand_map, output_dir, RAW_DATA_NAME
            )

    main_end_time = time.time()
    print(
        f"\n--- Анализ (v6 - Обогащение) ПОЛНОСТЬЮ завершен за {main_end_time - main_start_time:.2f} секунд ---"
    )
