# -*- coding: utf-8 -*-
"""
Скрипт для визуализации геометрических параметров взаимодействия
лиганд-ароматическая аминокислота (из aromatic_ring_geometry.csv).

Строит гистограммы, 2D scatter/density plot и box plots.
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Конфигурация ---
# Папки с результатами и для вывода
GEOMETRY_ANALYSIS_DIR = os.path.join(
    "analysis_results_v6_enriched", "geometry_analysis"
)
INPUT_CSV_FILE = os.path.join(GEOMETRY_ANALYSIS_DIR, "aromatic_ring_geometry.csv")
PLOT_OUTPUT_DIR = os.path.join(GEOMETRY_ANALYSIS_DIR, "plots") # Подпапка для графиков

# Параметры графиков
FIGSIZE_HIST = (10, 6)
FIGSIZE_SCATTER = (10, 8)
FIGSIZE_BOX = (8, 6)
SCATTER_ALPHA = 0.6
SCATTER_POINT_SIZE = 20
HIST_BINS = 50

# Параметры для аннотации стэкинга на Scatter Plot
STACKING_DIST_RANGE = (3.0, 5.5) # Типичное расстояние для стэкинга (можно уточнить)
PARALLEL_ANGLE_MAX = 30        # Макс угол для параллельного стэкинга
T_SHAPED_ANGLE_MIN = 70        # Мин угол для T-образного стэкинга

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

# --- Функции Построения Графиков ---

def plot_histogram(
    df,
    column,
    title,
    filename,
    hue_col=None,
    bins=HIST_BINS,
    figsize=FIGSIZE_HIST,
):
    """Строит гистограмму для указанной колонки, опционально с разбивкой по hue."""
    plt.figure(figsize=figsize)
    sns.histplot(
        data=df,
        x=column,
        hue=hue_col,
        bins=bins,
        kde=True,  # Оставляем KDE, оно тоже будет нормировано
        palette="viridis",
        stat="density",  # <--- ДОБАВИТЬ ЭТОТ ПАРАМЕТР
        common_norm=False  # <--- ВАЖНО: нормировать каждую группу (hue) отдельно!
    )
    # Изменить подпись оси Y:
    plt.ylabel("Плотность вероятности", fontsize=12)
    plt.title(title, fontsize=16)
    plt.xlabel(column.replace("_", " "), fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300)
        print(f"Гистограмма сохранена: {os.path.basename(filename)}")
    except Exception as e:
        print(f"! Ошибка сохранения гистограммы {os.path.basename(filename)}: {e}")
    plt.close()


def plot_scatter_angle_distance(
    df, filename, title, figsize=FIGSIZE_SCATTER
):
    """Строит 2D Scatter plot: Угол vs Расстояние, с аннотацией стэкинга."""
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=df,
        x="Normal_Angle_Deg",
        y="Centroid_Distance",
        hue="Neighbor_ResName",
        palette="deep",
        s=SCATTER_POINT_SIZE,
        alpha=SCATTER_ALPHA,
        edgecolor="w", # Небольшая обводка для лучшей видимости
        linewidth=0.5
    )

    # Аннотации для стэкинга
    # Область параллельного стэкинга
    ax.axvspan(0, PARALLEL_ANGLE_MAX, color='green', alpha=0.1, label=f'Параллельный (<{PARALLEL_ANGLE_MAX}°)')
    # Область T-образного стэкинга
    ax.axvspan(T_SHAPED_ANGLE_MIN, 90, color='blue', alpha=0.1, label=f'T-образный (>{T_SHAPED_ANGLE_MIN}°)')
    # Диапазон расстояний стэкинга
    ax.axhspan(STACKING_DIST_RANGE[0], STACKING_DIST_RANGE[1], color='grey', alpha=0.15, label=f'Расстояние {STACKING_DIST_RANGE} Å')

    # Добавляем текст для пояснения (опционально, может быть много)
    # ax.text(PARALLEL_ANGLE_MAX/2, STACKING_DIST_RANGE[1] * 1.05, 'Паралл.', ha='center', va='bottom', alpha=0.7)
    # ax.text((90+T_SHAPED_ANGLE_MIN)/2, STACKING_DIST_RANGE[1] * 1.05, 'T-образ.', ha='center', va='bottom', alpha=0.7)

    plt.title(title, fontsize=16)
    plt.xlabel("Угол между нормалями колец (°)", fontsize=12)
    plt.ylabel("Расстояние между центроидами колец (Å)", fontsize=12)
    plt.xlim(-5, 95) # Небольшие отступы по оси X
    # Установим разумные пределы для Y, например, до 8-10 Ангстрем
    max_dist = df["Centroid_Distance"].quantile(0.99) # 99-й перцентиль, чтобы обрезать выбросы
    plt.ylim(max(0, STACKING_DIST_RANGE[0] - 1), min(10, max_dist * 1.1))

    # Перемещаем легенду наружу
    plt.legend(title="Аминокислота", bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Оставляем место справа для легенды

    try:
        plt.savefig(filename, dpi=300)
        print(f"Scatter plot сохранен: {os.path.basename(filename)}")
    except Exception as e:
        print(f"! Ошибка сохранения scatter plot {os.path.basename(filename)}: {e}")
    plt.close()


def plot_2d_density(
    df, filename, title, figsize=FIGSIZE_SCATTER
):
    """Строит 2D Density plot: Угол vs Расстояние."""
    plt.figure(figsize=figsize)
    sns.kdeplot(
        data=df,
        x="Normal_Angle_Deg",
        y="Centroid_Distance",
        hue="Neighbor_ResName", # Можно раскрасить по типу АА
        palette="viridis", # Используем другую палитру
        fill=True,
        alpha=0.6,
        thresh=0.05, # Порог для отображения плотности
        bw_adjust=0.75 # Сглаживание
    )

    # Можно добавить контуры
    sns.kdeplot(
        data=df,
        x="Normal_Angle_Deg",
        y="Centroid_Distance",
        hue="Neighbor_ResName",
        palette="viridis",
        levels=5,
        linewidths=0.7
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Угол между нормалями колец (°)", fontsize=12)
    plt.ylabel("Расстояние между центроидами колец (Å)", fontsize=12)
    plt.xlim(-5, 95)
    max_dist = df["Centroid_Distance"].quantile(0.99)
    plt.ylim(max(0, STACKING_DIST_RANGE[0] - 1), min(10, max_dist * 1.1))
    plt.legend(title="Аминокислота", bbox_to_anchor=(1.03, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    try:
        plt.savefig(filename, dpi=300)
        print(f"Density plot сохранен: {os.path.basename(filename)}")
    except Exception as e:
        print(f"! Ошибка сохранения density plot {os.path.basename(filename)}: {e}")
    plt.close()


def plot_boxplots(
    df,
    x_col,
    y_col,
    title,
    filename,
    figsize=FIGSIZE_BOX,
):
    """Строит Box plot для сравнения распределений."""
    plt.figure(figsize=figsize)
    order = sorted(df[x_col].unique()) if x_col == 'Neighbor_ResName' else None
    sns.boxplot(data=df, x=x_col, y=y_col, palette="muted", order=order)
    # Можно добавить swarmplot для наглядности отдельных точек
    # sns.swarmplot(data=df, x=x_col, y=y_col, color=".25", size=3, alpha=0.5, order=order)

    plt.title(title, fontsize=16)
    plt.xlabel(x_col.replace("_", " "), fontsize=12)
    plt.ylabel(y_col.replace("_", " "), fontsize=12)
    plt.xticks(rotation=10) # Небольшой поворот, если нужно
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300)
        print(f"Box plot сохранен: {os.path.basename(filename)}")
    except Exception as e:
        print(f"! Ошибка сохранения box plot {os.path.basename(filename)}: {e}")
    plt.close()


# --- Основной Блок ---

if __name__ == "__main__":
    main_start_time = time.time()
    print("--- Визуализация Геометрии Взаимодействия Колец ---")
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    # 1. Загрузка данных
    print(f"\nШаг 1: Чтение данных геометрии из: {INPUT_CSV_FILE}")
    if not os.path.exists(INPUT_CSV_FILE):
        exit(f"! Ошибка: Файл с данными геометрии не найден: {INPUT_CSV_FILE}")
    try:
        df_geom = pd.read_csv(INPUT_CSV_FILE)
        # Уберем строки с NaN значениями, если они вдруг появились
        df_geom.dropna(subset=["Centroid_Distance", "Normal_Angle_Deg"], inplace=True)
        print(f"Загружено {len(df_geom)} валидных записей о геометрии.")
        if df_geom.empty:
             exit("! Нет валидных данных для визуализации.")
    except Exception as e:
        exit(f"! Ошибка чтения файла геометрии: {e}")

    # 2. Построение графиков
    print("\nШаг 2: Генерация графиков...")

    # Гистограммы
    plot_histogram(
        df_geom,
        "Centroid_Distance",
        "Распределение расстояний между центроидами колец",
        os.path.join(PLOT_OUTPUT_DIR, "hist_distance_overall.png"),
    )
    plot_histogram(
        df_geom,
        "Centroid_Distance",
        "Распределение расстояний (по типу аминокислоты)",
        os.path.join(PLOT_OUTPUT_DIR, "hist_distance_by_aa.png"),
        hue_col="Neighbor_ResName",
    )
    plot_histogram(
        df_geom,
        "Normal_Angle_Deg",
        "Распределение углов между нормалями колец",
        os.path.join(PLOT_OUTPUT_DIR, "hist_angle_overall.png"),
    )
    plot_histogram(
        df_geom,
        "Normal_Angle_Deg",
        "Распределение углов (по типу аминокислоты)",
        os.path.join(PLOT_OUTPUT_DIR, "hist_angle_by_aa.png"),
        hue_col="Neighbor_ResName",
    )

    # 2D Scatter Plot (Угол vs Расстояние)
    plot_scatter_angle_distance(
        df_geom,
        os.path.join(PLOT_OUTPUT_DIR, "scatter_angle_vs_distance.png"),
        "Геометрия взаимодействия: Угол vs Расстояние (цвет по АА)",
    )

    # 2D Density Plot (Угол vs Расстояние) - опционально
    plot_2d_density(
        df_geom,
        os.path.join(PLOT_OUTPUT_DIR, "density_angle_vs_distance.png"),
        "Плотность геометрий взаимодействия: Угол vs Расстояние (цвет по АА)",
    )

    # Box Plots для сравнения
    plot_boxplots(
        df_geom,
        x_col="Neighbor_ResName",
        y_col="Centroid_Distance",
        title="Сравнение расстояний для PHE/TYR/TRP",
        filename=os.path.join(PLOT_OUTPUT_DIR, "boxplot_distance_by_aa.png"),
    )
    plot_boxplots(
        df_geom,
        x_col="Neighbor_ResName",
        y_col="Normal_Angle_Deg",
        title="Сравнение углов для PHE/TYR/TRP",
        filename=os.path.join(PLOT_OUTPUT_DIR, "boxplot_angle_by_aa.png"),
    )
    # Можно добавить boxplot по лигандам, если их не слишком много
    if df_geom['Ligand_Code'].nunique() < 15: # Порог, чтобы график не был перегружен
         plot_boxplots(
             df_geom,
             x_col="Ligand_Code",
             y_col="Centroid_Distance",
             title="Сравнение расстояний по лигандам",
             filename=os.path.join(PLOT_OUTPUT_DIR, "boxplot_distance_by_ligand.png"),
             figsize=(max(FIGSIZE_BOX[0], df_geom['Ligand_Code'].nunique()*0.8), FIGSIZE_BOX[1]) # Делаем шире
         )
         plot_boxplots(
             df_geom,
             x_col="Ligand_Code",
             y_col="Normal_Angle_Deg",
             title="Сравнение углов по лигандам",
             filename=os.path.join(PLOT_OUTPUT_DIR, "boxplot_angle_by_ligand.png"),
             figsize=(max(FIGSIZE_BOX[0], df_geom['Ligand_Code'].nunique()*0.8), FIGSIZE_BOX[1])
         )

    main_end_time = time.time()
    print(f"\n--- Визуализация геометрии завершена за {main_end_time - main_start_time:.2f} секунд ---")
    print(f"Графики сохранены в папку: {PLOT_OUTPUT_DIR}")