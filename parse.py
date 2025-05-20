# -*- coding: utf-8 -*-
"""
Скрипт для поиска структур PDB по лигандам, сбора метаданных
(API + скрейпинг имен + GraphQL для TaxID/EMDB) и скачивания файлов.
"""

import concurrent.futures
import json
import os
import sys
import time
import traceback

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# --- Конфигурация ---
LIGAND_CODES = [
    "BCR",
    "ATX",
    "45D",
    "ECN",
    "ZEX",
    "LUT",
    "LUX",
    "A1L",
    "NEX",
    "XAT",
]
OUTPUT_CSV_FILE = "carotenoid_structures.csv"
CIF_DOWNLOAD_DIR = "pdb_cif_files"
EMDB_MAP_DOWNLOAD_DIR = "emdb_map_files"
DOWNLOAD_FILES = True
MAX_WORKERS = 8
REQUEST_TIMEOUT = 90
DOWNLOAD_TIMEOUT = 300
SCRAPE_DELAY = 0.5

# --- URL Шаблоны ---
SEARCH_API_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_API_URL = "https://data.rcsb.org/rest/v1/core/entry"
GRAPHQL_API_URL = "https://data.rcsb.org/graphql"
STRUCTURE_PAGE_URL_TEMPLATE = "https://www.rcsb.org/structure/{}"
CIF_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{}.cif"
EMDB_MAP_URL_TEMPLATE = (
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{}/map/{}.map.gz"
)


# --- Настройка сессии ---
def create_session_with_retries():
    """Создает сессию requests с повторами и User-Agent."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=None,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


session = create_session_with_retries()

# --- Функции ---


def search_pdb_by_ligand(ligand_code):
    """Ищет PDB ID по коду лиганда."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                "operator": "exact_match",
                "value": ligand_code,
                "negation": False,
            },
        },
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
        },
        "return_type": "entry",
    }
    response = None
    try:
        response = session.post(SEARCH_API_URL, json=query, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        results = response.json()
        pdb_ids = [item["identifier"] for item in results.get("result_set", [])]
        return ligand_code, pdb_ids
    except Exception:
        return ligand_code, []


def get_entry_details(pdb_id):
    """Получает детальную информацию из Data API."""
    url = f"{DATA_API_URL}/{pdb_id}"
    response = None
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_organism_via_scraping(pdb_id):
    """Скрейпит имя организма со страницы RCSB."""
    url = STRUCTURE_PAGE_URL_TEMPLATE.format(pdb_id)
    try:
        time.sleep(SCRAPE_DELAY)
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        parser_type = "lxml"
        try:
            __import__("lxml")
        except ImportError:
            parser_type = "html.parser"
        soup = BeautifulSoup(response.content, parser_type)
        organism_link = soup.select_one("li#header_organism > a")
        if organism_link:
            organism_text = organism_link.get_text(strip=True)
            if organism_text:
                return organism_text
        return "N/A"
    except Exception:
        return "N/A"


def get_info_via_graphql(pdb_id):
    """Получает EMDB ID и TaxID с помощью GraphQL."""
    query = """
    query($entry_id: String!){
      entry(entry_id: $entry_id) {
        rcsb_entry_container_identifiers {
          emdb_ids
        }
        polymer_entities {
          rcsb_entity_source_organism {
            ncbi_taxonomy_id
          }
        }
      }
    }
    """
    variables = {"entry_id": pdb_id}
    graphql_result = {"emdb_id": None, "tax_ids": []}
    try:
        response = session.post(
            GRAPHQL_API_URL,
            json={"query": query, "variables": variables},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if data and "data" in data and data["data"].get("entry"):
            entry_data = data["data"]["entry"]

            # Извлечение EMDB ID
            container_ids = entry_data.get("rcsb_entry_container_identifiers")
            if isinstance(container_ids, dict):
                emdb_ids = container_ids.get("emdb_ids")
                if isinstance(emdb_ids, list) and emdb_ids:
                    first_id = emdb_ids[0]
                    if isinstance(
                        first_id, str
                    ) and first_id.strip().upper().startswith("EMD-"):
                        graphql_result["emdb_id"] = first_id.strip().upper()

            # Извлечение Tax ID
            polymer_entities = entry_data.get("polymer_entities")
            if isinstance(polymer_entities, list):
                tax_id_set = set()
                for entity in polymer_entities:
                    if isinstance(entity, dict):
                        source_orgs = entity.get("rcsb_entity_source_organism")
                        if isinstance(source_orgs, list):
                            for org in source_orgs:
                                if isinstance(org, dict):
                                    tax_id = org.get("ncbi_taxonomy_id")
                                    if tax_id is not None:
                                        tax_id_set.add(str(tax_id))
                if tax_id_set:
                    graphql_result["tax_ids"] = sorted(list(tax_id_set))

        return graphql_result

    except Exception as e:
        # print(f"GraphQL ERROR for {pdb_id}: {e}") # Отладка
        return graphql_result  # Возвращаем дефолтный словарь при ошибке


def parse_details(data, pdb_id, ligand_code):
    """Извлекает основные поля из ответа Data API (БЕЗ TaxID и EMDB)."""
    if not isinstance(data, dict):
        return None

    # Инициализация (TaxID и EMDB будут заполнены позже из GraphQL)
    result = {
        "pdbID": pdb_id,
        "Ligand": ligand_code,
        "Method": "N/A",
        "Resolution": "N/A",
        "Organism_Name": "N/A",
        "Organism_TaxID": "N/A",
        "EMDB_for_EMRelease": "N/A",
        "Pub_Title": "N/A",
        "Pub_Authors": "N/A",
        "Pub_Journal": "N/A",
        "Pub_Year": "N/A",
        "Pub_PubMed": "N/A",
        "Pub_DOI": "N/A",
        "Coord_File": f"{pdb_id}.cif" if pdb_id else "N/A",
        "EM_Map_File": "N/A",
    }
    try:
        # Метод и разрешение
        exptl = data.get("exptl", [{}])[0]
        if isinstance(exptl, dict):
            result["Method"] = exptl.get("method", "N/A")
        entry_info = data.get("rcsb_entry_info")
        if isinstance(entry_info, dict):
            res_list = entry_info.get("resolution_combined")
            if isinstance(res_list, list) and res_list and res_list[0] is not None:
                result["Resolution"] = str(res_list[0])

        # Имя организма (из API)
        organisms_data = data.get("rcsb_entity_source_organism")
        org_names = []
        if isinstance(organisms_data, list):
            for o in organisms_data:
                if isinstance(o, dict):
                    name = o.get("ncbi_scientific_name")
                    if isinstance(name, str) and name:
                        org_names.append(name)
        if org_names:
            result["Organism_Name"] = "; ".join(sorted(list(set(org_names))))

        # Публикации (из API)
        citation_data = None
        citation_list = data.get("citation")
        if isinstance(citation_list, list) and citation_list:
            citation_data = citation_list[0]
        if not isinstance(citation_data, dict):
            citation_data = data.get("rcsb_primary_citation")
        if isinstance(citation_data, dict):
            result["Pub_PubMed"] = str(
                citation_data.get("pdbx_database_id_PubMed", "N/A")
            )
            result["Pub_DOI"] = citation_data.get("pdbx_database_id_DOI", "N/A")
            result["Pub_Journal"] = citation_data.get(
                "journal_abbrev", citation_data.get("rcsb_journal_abbreviation", "N/A")
            )
            result["Pub_Year"] = str(citation_data.get("year", "N/A"))
            result["Pub_Title"] = citation_data.get("title", "N/A")
            authors = citation_data.get("rcsb_authors")
            if not authors:
                audit_authors_list = data.get("audit_author")
                if isinstance(audit_authors_list, list):
                    authors = [
                        a.get("name")
                        for a in audit_authors_list
                        if isinstance(a, dict) and a.get("name")
                    ]
            result["Pub_Authors"] = (
                ", ".join(authors) if isinstance(authors, list) and authors else "N/A"
            )

        # Очистка пустых значений
        for key, value in result.items():
            if value is None or str(value).strip() == "":
                result[key] = "N/A"
        return result

    except Exception as e:
        print(f"ОШИБКА ПАРСИНГА API для {pdb_id}: {e}")
        traceback.print_exc()
        result["Method"] = "API Parse Error"
        return result


def fetch_parse_and_scrape(pdb_id_ligand_tuple):
    """Оркестратор: получает данные API, парсит, запрашивает GraphQL, скрейпит."""
    pdb_id, ligand_code = pdb_id_ligand_tuple
    api_data = get_entry_details(pdb_id)
    parsed_data = None

    if api_data:
        parsed_data = parse_details(api_data, pdb_id, ligand_code)

    # Инициализация базового словаря, если парсинг API не удался
    if parsed_data is None:
        parsed_data = {
            "pdbID": pdb_id,
            "Ligand": ligand_code,
            "Method": "API Fetch Failed",
            "Resolution": "N/A",
            "Organism_Name": "N/A",
            "Organism_TaxID": "N/A",
            "EMDB_for_EMRelease": "N/A",
            "Pub_Title": "N/A",
            "Pub_Authors": "N/A",
            "Pub_Journal": "N/A",
            "Pub_Year": "N/A",
            "Pub_PubMed": "N/A",
            "Pub_DOI": "N/A",
            "Coord_File": f"{pdb_id}.cif",
            "EM_Map_File": "N/A",
        }

    # --- Получаем TaxID и EMDB ID через GraphQL ---
    graphql_info = get_info_via_graphql(pdb_id)
    if graphql_info:
        # Обновляем TaxID
        if graphql_info.get("tax_ids"):
            parsed_data["Organism_TaxID"] = "; ".join(graphql_info["tax_ids"])
        # Обновляем EMDB ID и имя файла карты
        if graphql_info.get("emdb_id"):
            emdb_id = graphql_info["emdb_id"]
            parsed_data["EMDB_for_EMRelease"] = emdb_id
            emdb_id_lower = emdb_id.lower().replace("-", "_")
            parsed_data["EM_Map_File"] = f"{emdb_id_lower}.map.gz"

    # --- Скрейпим Имя организма, если оно не получено из API ---
    if parsed_data.get("Organism_Name") == "N/A":
        scraped_organism_name = get_organism_via_scraping(pdb_id)
        if scraped_organism_name != "N/A":
            parsed_data["Organism_Name"] = scraped_organism_name

    # Финальная проверка на N/A для всех полей
    for key, value in parsed_data.items():
        if value is None or str(value).strip() == "":
            parsed_data[key] = "N/A"

    return parsed_data


def download_file(url, filepath, description, file_type):
    """Скачивает файл (CIF или MAP)."""
    if os.path.exists(filepath):
        return description, True
    response = None
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        response = session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return description, True
    except Exception:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        return description, False


# --- Основной скрипт ---
def main():
    """Главная функция выполнения скрипта."""
    start_time = time.time()

    # === Шаг 1: Поиск PDB ID ===
    print(f"Шаг 1: Поиск PDB ID для {len(LIGAND_CODES)} лигандов...")
    ligand_to_pdb_ids = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS, thread_name_prefix="Search"
    ) as executor:
        futures_search = {
            executor.submit(search_pdb_by_ligand, lc): lc for lc in LIGAND_CODES
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures_search),
            total=len(LIGAND_CODES),
            desc="Поиск лигандов",
        ):
            ligand_code = futures_search[future]
            try:
                _, pdb_ids = future.result()
                if pdb_ids:
                    ligand_to_pdb_ids[ligand_code] = pdb_ids
            except Exception as exc:
                print(f"Error in search future {ligand_code}: {exc}")

    tasks_fetch = []
    unique_pdb_ids_found = set()
    for ligand, ids in ligand_to_pdb_ids.items():
        for pdb_id in ids:
            tasks_fetch.append((pdb_id, ligand))
            unique_pdb_ids_found.add(pdb_id)

    if not tasks_fetch:
        print("\nСтруктуры не найдены.")
        sys.exit()
    print(
        f"\nНайдено {len(unique_pdb_ids_found)} уник. PDB ID. "
        f"Всего записей для обработки: {len(tasks_fetch)}"
    )

    # === Шаг 2: Получение/Парсинг/Скрейпинг/GraphQL ===
    print("\nШаг 2: Получение/Парсинг деталей (API + GraphQL + Скрейпинг)...")
    all_data = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS, thread_name_prefix="ProcessEntry"
    ) as executor:
        futures_proc = {
            executor.submit(fetch_parse_and_scrape, task): task for task in tasks_fetch
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures_proc),
            total=len(tasks_fetch),
            desc="Обработка записей",
        ):
            task_info = futures_proc[future]
            try:
                result = future.result()
                if result is not None:
                    all_data.append(result)
            except Exception as exc:
                print(f"Critical error processing task {task_info}: {exc}")

    print(f"\nСобрано {len(all_data)} записей с деталями.")

    # === Шаг 3: Сохранение в CSV ===
    if all_data:
        df = pd.DataFrame(all_data)
        cols_order = [
            "pdbID",
            "Ligand",
            "Method",
            "Resolution",
            "Organism_Name",
            "Organism_TaxID",
            "EMDB_for_EMRelease",
            "Pub_Title",
            "Pub_Authors",
            "Pub_Journal",
            "Pub_Year",
            "Pub_PubMed",
            "Pub_DOI",
            "Coord_File",
            "EM_Map_File",
        ]
        for col in cols_order:
            if col not in df.columns:
                df[col] = "N/A"
        df = df[cols_order]
        df = df.drop_duplicates()
        df.fillna("N/A", inplace=True)
        df.replace("", "N/A", inplace=True)
        df.sort_values(by=["Ligand", "pdbID"], inplace=True)
        try:
            df.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")
            print(f"\nШаг 3: Данные ({len(df)} строк) сохранены: {OUTPUT_CSV_FILE}")
        except Exception as e:
            print(f"\nОшибка сохранения CSV: {e}")
    else:
        print("\nНет успешно собранных данных для сохранения в CSV.")
        df = pd.DataFrame()

    # === Шаг 4: Скачивание файлов ===
    if DOWNLOAD_FILES and not df.empty:
        print("\nШаг 4: Скачивание файлов...")
        coord_tasks = []
        map_tasks = []

        # Задачи на скачивание координат (CIF)
        coord_data = df[df["Coord_File"] != "N/A"][
            ["pdbID", "Coord_File"]
        ].drop_duplicates()
        for _, row in coord_data.iterrows():
            pdb_id, filename = row["pdbID"], row["Coord_File"]
            if pdb_id != "N/A" and filename != "N/A":
                url = CIF_DOWNLOAD_URL_TEMPLATE.format(pdb_id)
                fpath = os.path.join(CIF_DOWNLOAD_DIR, filename)
                coord_tasks.append((url, fpath, filename, "cif"))

        # # Задачи на скачивание карт EMDB (MAP.GZ)
        # map_data = df[
        #     (df['EMDB_for_EMRelease'] != 'N/A') & (df['EM_Map_File'] != 'N/A')
        # ][['EMDB_for_EMRelease', 'EM_Map_File']].drop_duplicates()
        # for _, row in map_data.iterrows():
        #     emdb_id, filename = row['EMDB_for_EMRelease'], row['EM_Map_File']
        #     if (emdb_id and isinstance(emdb_id, str)
        #             and emdb_id.upper().startswith("EMD-") and filename != 'N/A'):
        #         emdb_id_upper = emdb_id.upper()
        #         emdb_id_lower_url = emdb_id_upper.lower().replace('-', '_')
        #         url = EMDB_MAP_URL_TEMPLATE.format(emdb_id_upper, emdb_id_lower_url)
        #         fpath = os.path.join(EMDB_MAP_DOWNLOAD_DIR, filename)
        #         map_tasks.append((url, fpath, filename, "map"))

        print(f"  Задач на скачивание координат (CIF): {len(coord_tasks)}")
        print(f"  Задач на скачивание карт EMDB: {len(map_tasks)}")  # ВАЖНО!

        all_tasks = coord_tasks + map_tasks
        if all_tasks:
            success_count, fail_count = 0, 0
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS, thread_name_prefix="Download"
            ) as executor:
                futures_dl = {
                    executor.submit(download_file, url, fp, desc, ftype): desc
                    for url, fp, desc, ftype in all_tasks
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures_dl),
                    total=len(all_tasks),
                    desc="Скачивание файлов",
                ):
                    desc = futures_dl[future]
                    try:
                        _, success = future.result()
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as exc:
                        print(f"Error processing download future for {desc}: {exc}")
                        fail_count += 1
            print(
                f"\nСкачивание завершено. Успешно/Существовало: {success_count}, "
                f"Ошибок/Пропущено: {fail_count}"
            )
            print(f"Координатные файлы (CIF) сохранены в: {CIF_DOWNLOAD_DIR}")
            print(f"Карты EMDB сохранены в: {EMDB_MAP_DOWNLOAD_DIR}")
        else:
            print("Нет файлов для скачивания.")

    # === Завершение ===
    end_time = time.time()
    print(f"\nОбщее время выполнения: {end_time - start_time:.2f} секунд.")


if __name__ == "__main__":
    # Проверка и предупреждение об отсутствии lxml
    try:
        import lxml  # noqa: F401
    except ImportError:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: lxml не найден, используется html.parser. "
            "'pip install lxml' для ускорения."
        )
    # Запуск основной функции
    main()
