#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Moh'd Khier Al Kfari"
__copyright__ = "Copyright (C) 2025 Moh'd Khier Al Kfari"

"""
Interactive preprocessor for the DaRA dataset annotations.

Key Features
-------------
- **Automatic discovery** of available annotation categories from 'Final__Annotation__CCxx_*' folders.
- **Interactive selection** of which annotation types to include (or choose 'All').
- **Scheme-aware label extraction:**
  - Uses the corresponding JSON scheme files to correctly interpret and map annotation codes (CLxxx) to labels.
  - Handles both single-label and multi-label cases efficiently using vectorized, column-wise operations.
- **Special structure handling:**
  - **Location annotations** are automatically split into two columns:
      - 'Location – Human (Main)' / 'Location – Human (Sub)'
      - 'Location – Cart (Main)' / 'Location – Cart (Sub)'
    Sub-locations are parsed based on scheme hierarchy (e.g., Path, Cross Aisle Path, Aisle Path).
  - **Hand annotations (Left/Right)** are automatically expanded into four parts:
      - 'Primary Position', 'Type of Movement', 'Object', and 'Tool'.
- **Interactive combination options:**
  - Optionally create 'input' (e.g., activities + sub-activities + locations + IT + order)
    and/or 'output' (e.g., high/mid/low-level process) columns.
  - For each selected Location or Hand annotation, you can interactively choose which subparts
    (e.g., Main/Sub for Locations or Primary Position/Object/Tool for Hands) to include in the combinations.
- **Filtering options:**
  - Optionally remove entries containing "Unknown" or "Another/Other" labels (based on scheme definitions).
- **Merging:**
  - Merges all selected annotations **by row index**.
- **Data integrity:**
  - Missing or unavailable labels are replaced with "-".
- **Output:**
  - Saves per-subject preprocessed CSVs (`S01.csv` ... `S18.csv`) under:
    `'Annotation_Rostock/data_preprocessed/'`.

Assumptions
------------
- Each per-category CSV has the first column as a numeric or parseable timestamp
  (ignored during merging, since all data streams are time-aligned and synchronized).
- The folder structure follows the DaRA annotation convention with corresponding `scheme__CCxx_*.json` files.
"""

import os
import re
import glob
import json
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import zipfile
import shutil
from pathlib import Path

# ----------------------------- Paths ---------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
SCHEME_DIR = os.path.join(RAW_DIR, "scheme")
OUT_DIR = os.path.join(BASE_DIR, "data_preprocessed")

# Default subject list (S01..S18)
SUBJECTS = [f"S{i:02}" for i in range(1, 19)]

# Canonical “families” for convenience (user still explicitly chooses)
INPUT_FAMILY = [
    "Main Activity",
    "Sub-Activity – Legs",
    "Sub-Activity – Torso",
    "Sub-Activity – Left Hand",
    "Sub-Activity – Right Hand",
    "Location – Human",
    "Location – Cart",
    "Information Technology",
    "Order",
]
OUTPUT_FAMILY = [
    "High-Level Process",
    "Mid-Level Process",
    "Low-Level Process",
]



def _move_children_up(src_dir: str, dst_dir: str) -> None:
    """
    If the zip extracts to raw_data/raw_data/*, move the inner children up into raw_data/.
    """
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        if os.path.exists(dst_path):
            # merge folders; if conflict, prefer existing to avoid accidental overwrite
            if os.path.isdir(src_path) and os.path.isdir(dst_path):
                for child in os.listdir(src_path):
                    csrc = os.path.join(src_path, child)
                    cdst = os.path.join(dst_path, child)
                    if not os.path.exists(cdst):
                        shutil.move(csrc, cdst)
            else:
                # if a file already exists, skip moving it
                continue
        else:
            shutil.move(src_path, dst_path)


def ensure_raw_data_ready() -> None:
    """
    Ensure RAW_DIR exists and contains the annotation folders.
    If RAW_DIR is missing or empty, attempt to extract 'raw_data.zip'
    from the same directory as this script.

    Supported cases:
      - Zip contains the folder contents directly (Final__Annotation__CCxx_* and scheme/)
      - Zip contains a top-level 'raw_data/' folder (we flatten it)

    Safe behavior:
      - If RAW_DIR already has data, we do nothing.
      - If RAW_DIR exists but is empty, we extract.
    """
    # If a non-empty raw_data folder already exists, we're done.
    if os.path.isdir(RAW_DIR) and any(os.scandir(RAW_DIR)):
        return

    # Find a zip named 'raw_data.zip' in the same folder as this script.
    zip_candidates = [
        os.path.join(BASE_DIR, "raw_data.zip"),
        os.path.join(BASE_DIR, "raw_data.ZIP"),
    ]
    zip_path = next((p for p in zip_candidates if os.path.isfile(p)), None)

    if not zip_path:
        # Nothing to do; leave detection to fail later with a clear message
        print("⚠ Could not find 'raw_data/' or 'raw_data.zip' next to the script.")
        return

    print(f"📦 Extracting {os.path.basename(zip_path)} ...")
    os.makedirs(RAW_DIR, exist_ok=True)

    # Extract into a temporary folder first to handle nested layouts safely
    tmp_extract_dir = os.path.join(BASE_DIR, "_tmp_raw_extract")
    if os.path.isdir(tmp_extract_dir):
        shutil.rmtree(tmp_extract_dir)
    os.makedirs(tmp_extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_extract_dir)

    # If the zip already contains the expected folders, move them into RAW_DIR.
    # Handle the "raw_data/*" wrapping if present.
    # Find a likely top-level that holds the Final__Annotation__* and scheme/*.
    # Strategy:
    #   - If tmp has a single directory, descend into it.
    #   - Repeat once more if needed (covers raw_data/raw_data/*)
    probe = tmp_extract_dir
    for _ in range(2):
        entries = [p for p in os.listdir(probe) if not p.startswith(".")]
        if len(entries) == 1 and os.path.isdir(os.path.join(probe, entries[0])):
            probe = os.path.join(probe, entries[0])
        else:
            break

    # Now move children of `probe` into RAW_DIR
    _move_children_up(probe, RAW_DIR)

    # Clean up temp
    shutil.rmtree(tmp_extract_dir, ignore_errors=True)
    print(f"✅ Extracted into: {RAW_DIR}")

# ----------------------------- Discovery & Schemes -------------------------------

def detect_annotation_folders(raw_dir: str) -> Dict[str, str]:
    """
    Purpose:
        Discover available annotation categories by scanning 'Final__Annotation__CCxx_*' folders.

    Args:
        raw_dir (str): Absolute or relative path to 'raw_data/'.

    Returns:
        Dict[str, str]:
            Mapping {friendly_category_name -> absolute_path_to_category_folder}.
            Example: {"Main Activity": ".../Final__Annotation__CC01_Main Activity"}
    """
    mapping = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "Final__Annotation__CC*"))):
        if not os.path.isdir(path):
            continue
        tail = os.path.basename(path)
        m = re.search(r"CC\d+_(.+)$", tail)  # extract the part after CCxx_
        if not m:
            continue
        friendly = m.group(1).strip()
        mapping[friendly] = os.path.abspath(path)
    return mapping


def read_scheme_files_with_raw(scheme_dir: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, set], Dict[str, set], Dict[str, Any]]:
    """
    Purpose:
        Load JSON scheme files and build:
        - code_to_label per scheme (e.g., "CL001" -> "Synchronization")
        - 'unknown' label sets per scheme
        - 'another/other' label sets per scheme
        - raw scheme JSON content for hierarchy-aware extraction

    Args:
        scheme_dir (str): Path to the folder containing 'scheme__*.json' files.

    Returns:
        (code_maps, unknown_sets, other_sets, raw_by_key)
        code_maps: Dict[str, Dict[str, str]]
            { scheme_base_name -> { code -> label } }
        unknown_sets: Dict[str, set]
            { scheme_base_name -> set of labels that indicate Unknown }
        other_sets: Dict[str, set]
            { scheme_base_name -> set of labels that indicate Another/Other }
        raw_by_key: Dict[str, Any]
            { scheme_base_name -> parsed JSON list of [Group, [ "CLxxx|Label", ... ]] }
    """
    code_maps: Dict[str, Dict[str, str]] = {}
    unknown_sets: Dict[str, set] = {}
    other_sets: Dict[str, set] = {}
    raw_by_key: Dict[str, Any] = {}

    for f in glob.glob(os.path.join(scheme_dir, "scheme__*.json")):
        base = os.path.basename(f)
        scheme_name = base.replace("scheme__", "").replace(".json", "")
        code_maps[scheme_name] = {}
        unknown_sets[scheme_name] = set()
        other_sets[scheme_name] = set()

        try:
            data = json.load(open(f, "r", encoding="utf-8"))
            raw_by_key[scheme_name] = data
        except Exception:
            raw_by_key[scheme_name] = []
            continue

        # 'data' is a list of [GroupName, [ "CLxxx|Label", ... ]]
        for section in data:
            if not isinstance(section, list) or len(section) != 2:
                continue
            _, items = section
            for item in items:
                if not isinstance(item, str) or "|" not in item:
                    continue
                code, label = item.split("|", 1)
                code = code.strip()
                label = label.strip()
                code_maps[scheme_name][code] = label

                low = label.lower()
                if "unknown" in low:
                    unknown_sets[scheme_name].add(label)
                if "another" in low or "other" in low:
                    other_sets[scheme_name].add(label)

    return code_maps, unknown_sets, other_sets, raw_by_key

# ----------------------------- CLI Helpers -------------------------------

def print_menu(options: List[str]) -> None:
    """
    Purpose:
        Pretty-print a numbered menu for the user.

    Args:
        options (List[str]): Options to print.

    Returns:
        None
    """
    print("\nAvailable annotation types:")
    for i, name in enumerate(options, start=1):
        print(f"{i:>2}. {name}")
    print(f"{len(options)+1:>2}. All")


def choose_from_menu(options: List[str]) -> List[str]:
    """
    Purpose:
        Present a numbered menu and let the user select multiple options or 'All'.
        The user can type numbers separated by commas or spaces (flexible parsing).

    Args:
        options (List[str]): List of available annotation types.

    Returns:
        List[str]: The selected annotation names.
    """
    print_menu(options)
    while True:
        ans = input("\nEnter the numbers of annotation types you want (e.g., 1,2,3 or 'all'): ").strip().lower()

        # handle 'all' regardless of case or spaces
        if ans.replace(" ", "") == "all":
            chosen = list(options)
            print("\n✅ Selected ALL annotation types:")
            for c in chosen:
                print(f"   - {c}")
            return chosen

        cleaned = re.sub(r"[ ,]+", ",", ans.strip(", "))  # normalize separators/spaces
        if not cleaned:
            print("✖ No input detected. Please enter numbers or 'all'.")
            continue

        try:
            indices = [int(x) for x in cleaned.split(",") if x.strip().isdigit()]
            chosen = []
            for idx in indices:
                if 1 <= idx <= len(options):
                    chosen.append(options[idx - 1])
                elif idx == len(options) + 1:
                    chosen = list(options)
                    break
                else:
                    print(f"⚠ Index {idx} is out of range.")
            if not chosen:
                print("✖ No valid indices found. Try again.")
                continue

            # de-dup while preserving order
            chosen = list(dict.fromkeys(chosen))

            print("\n✅ Selected annotation types:")
            for c in chosen:
                print(f"   - {c}")
            return chosen

        except Exception as e:
            print(f"✖ Invalid input ({e}). Try again.")


def choose_combo_columns(available_columns: List[str], title: str) -> List[str]:
    """
    Purpose:
        Ask the user which of the available columns should be combined for a combo column.

    Args:
        available_columns (List[str]): Candidate columns for combination (already filtered to what's in the merged table).
        title (str): Heading to display (e.g., "INPUT combination" or "OUTPUT combination").

    Returns:
        List[str]: Columns (in user-selected order) to combine. If user selects 'All', returns all candidates.
    """
    print(f"\nSelect which columns to combine for the {title}:")
    selected = choose_from_menu(available_columns)
    return selected


def choose_location_parts(location_label: str) -> List[str]:
    """
    Purpose:
        Ask which parts of a location category to use in the input combo (Main/Sub).

    Args:
        location_label (str): Either 'Location – Human' or 'Location – Cart'.

    Returns:
        List[str]: Subset of ['Main', 'Sub'].
                   If user selects both (via 'all' or '3' or '1,2' etc.), returns ['Main', 'Sub'] and
                   prints a unified message: "✅ Using both Main and Sub for <location_label>."
    """
    options = ["Main", "Sub"]
    print(f"\n{location_label} is included in the INPUT combination.")
    print("Which parts do you want to use?")
    for i, name in enumerate(options, start=1):
        print(f"{i:>2}. {name} location")
    print(f"{len(options)+1:>2}. All")

    while True:
        ans = input("Enter numbers for Main/Sub (e.g., 1,2 or 'all'): ").strip().lower()

        # 'all' (any casing/spaces) -> both
        if ans.replace(" ", "") == "all":
            print(f"✅ Using both Main and Sub for {location_label}.")
            return ["Main", "Sub"]

        # Normalize separators/spaces and parse indices
        cleaned = re.sub(r"[ ,]+", ",", ans.strip(", "))
        if not cleaned:
            print("✖ No input detected. Please enter numbers or 'all'.")
            continue

        try:
            idxs = [int(x) for x in cleaned.split(",") if x.strip().isdigit()]
            # Map indices to parts; accept 3 as 'All'
            chosen_parts = []
            saw_all = False
            for idx in idxs:
                if idx == 3:  # All
                    saw_all = True
                    break
                elif idx == 1:
                    chosen_parts.append("Main")
                elif idx == 2:
                    chosen_parts.append("Sub")
                else:
                    print(f"⚠ Index {idx} is out of range (valid: 1, 2, 3).")

            if saw_all:
                print(f"✅ Using both Main and Sub for {location_label}.")
                return ["Main", "Sub"]

            # Deduplicate while preserving order
            chosen_parts = list(dict.fromkeys(chosen_parts))

            if not chosen_parts:
                print("✖ No valid indices found. Try again.")
                continue

            # If both selected, normalize order and message
            if set(chosen_parts) == {"Main", "Sub"}:
                print(f"✅ Using both Main and Sub for {location_label}.")
                return ["Main", "Sub"]

            # Otherwise, single selection remains
            print(f"✅ Selected parts for {location_label}: " + ", ".join(chosen_parts))
            return chosen_parts

        except Exception as e:
            print(f"✖ Invalid input ({e}). Try again.")

def choose_hand_parts(hand_label: str) -> List[str]:
    """
    Purpose:
        Ask which parts of a hand category to use in the INPUT combo.

    Args:
        hand_label (str): 'Sub-Activity – Left Hand' or 'Sub-Activity – Right Hand'.

    Returns:
        List[str]: Subset of ['Primary Position','Type of Movement','Object','Tool'].
                   If user selects All (via 'all', '5', or all indices), returns all four.
    """
    options = ["Primary Position", "Type of Movement", "Object", "Tool"]
    print(f"\n{hand_label} is included in the INPUT combination.")
    print("Which parts do you want to use?")
    for i, name in enumerate(options, start=1):
        print(f"{i:>2}. {name}")
    print(f"{len(options)+1:>2}. All")

    while True:
        ans = input("Enter numbers for parts (e.g., 1,2 or 'all'): ").strip().lower()
        if ans.replace(" ", "") == "all":
            print(f"✅ Using all parts for {hand_label}.")
            return options[:]

        cleaned = re.sub(r"[ ,]+", ",", ans.strip(", "))
        if not cleaned:
            print("✖ No input detected. Please enter numbers or 'all'.")
            continue

        try:
            idxs = [int(x) for x in cleaned.split(",") if x.strip().isdigit()]
            chosen_parts, saw_all = [], False
            for idx in idxs:
                if idx == len(options) + 1:
                    saw_all = True
                    break
                if 1 <= idx <= len(options):
                    chosen_parts.append(options[idx - 1])
                else:
                    print(f"⚠ Index {idx} is out of range (valid: 1..{len(options)+1}).")

            if saw_all:
                print(f"✅ Using all parts for {hand_label}.")
                return options[:]

            chosen_parts = list(dict.fromkeys(chosen_parts))
            if not chosen_parts:
                print("✖ No valid indices found. Try again.")
                continue

            # If effectively all were selected in any order
            if set(chosen_parts) == set(options):
                print(f"✅ Using all parts for {hand_label}.")
                return options[:]

            print(f"✅ Selected parts for {hand_label}: " + ", ".join(chosen_parts))
            return chosen_parts

        except Exception as e:
            print(f"✖ Invalid input ({e}). Try again.")

def ask_yes_no(prompt: str) -> bool:
    """
    Purpose:
        Ask a yes/no question and only accept 'y' or 'n' as valid input.

    Args:
        prompt (str): The question to show.

    Returns:
        bool: True if the user answered 'y', False if 'n'.
    """
    while True:
        answer = input(f"{prompt} (y/n): ").strip().lower()
        if answer in ["y", "n"]:
            return answer == "y"
        print("✖ Invalid input. Please enter 'y' or 'n' only.")

# ----------------------------- IO Helpers -------------------------------

def read_subject_tables(category_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Purpose:
        Read all subject CSVs under a given category.

    Args:
        category_folder (str): Path like '.../Final__Annotation__CC01_Main Activity'.

    Returns:
        Dict[str, pd.DataFrame]:
            Mapping {'S01': df, ...}. The DataFrame keeps all original columns.
            Assumes the first column is the timestamp (ignored later).
    """
    out: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(glob.glob(os.path.join(category_folder, "*.csv"))):
        base = os.path.basename(csv_path)
        m = re.search(r"__S(\d{2})\.csv$", base)
        if not m:
            m = re.search(r"S(\d{2})", base)
        if not m:
            continue
        subj = f"S{m.group(1)}"
        df = pd.read_csv(csv_path)
        if df.shape[1] == 0:
            continue

        # keep as-is; timestamp is column 0 but we won't use it for merging
        out[subj] = df
    return out

# ======================= Scheme-aware fast extractor =======================

def parse_scheme(scheme_json: Any) -> Dict[str, Any]:
    """
    Purpose:
        Parse a scheme JSON into structured groups and extract semantic hints for Locations:
        - aisle_numbers: labels that are pure numbers ("1","2",...)
        - aisle_sides: 'Front'/'Back'
        - path_specific: labels like 'Path (Office)'
        - cross_pairs: labels like '1-2', '2-3'
        Also flags for hand/location semantics.

    Args:
        scheme_json: Parsed JSON from 'scheme__*.json'.

    Returns:
        Dict with keys:
          - groups: List[Dict] each with:
              { "name": str, "labels": [{"code","label"}...], "label_names": [str,...] }
          - has_location_semantics: bool
          - has_hand_semantics: bool
          - aisle_numbers: set[str]
          - aisle_sides: set[str]
          - path_specific: set[str]
          - cross_pairs: set[str]
    """
    groups = []
    aisle_numbers, aisle_sides = set(), set()
    path_specific, cross_pairs = set(), set()

    def is_number_token(s: str) -> bool:
        return bool(re.fullmatch(r"\d+", s.strip()))

    def is_side_token(s: str) -> bool:
        return s.strip().lower() in {"front", "back"}

    def is_path_specific(s: str) -> bool:
        return bool(re.match(r"^Path\s*\(", s.strip(), flags=re.IGNORECASE))

    def is_cross_pair(s: str) -> bool:
        return bool(re.fullmatch(r"\d+\s*-\s*\d+", s.strip()))

    for entry in scheme_json or []:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        group_name, items = entry
        parsed_items, label_names = [], []
        for item in items:
            if not isinstance(item, str) or "|" not in item:
                continue
            code, label = item.split("|", 1)
            code = code.strip()
            label = label.strip()
            parsed_items.append({"code": code, "label": label})
            label_names.append(label)

            # collect hints
            if is_number_token(label):
                aisle_numbers.add(label)
            if is_side_token(label):
                aisle_sides.add(label)
            if is_path_specific(label):
                path_specific.add(label)
            if is_cross_pair(label):
                cross_pairs.add(label)

        groups.append({"name": group_name, "labels": parsed_items, "label_names": label_names})

    # Semantics flags
    text_all = " ".join([g["name"] for g in groups] + sum([g["label_names"] for g in groups], []))
    has_location = (
        ("location" in text_all.lower()) or
        any(
            any(k in g["name"].lower() for k in ["aisle", "path", "area"])
            for g in groups
        )
    )
    has_hand = any(
        ("hand" in g["name"].lower()) or
        ("primary position" in g["name"].lower()) or
        ("type of movement" in g["name"].lower()) or
        ("object" in g["name"].lower()) or
        ("tool" in g["name"].lower())
        for g in groups
    )

    return {
        "groups": groups,
        "has_location_semantics": has_location,
        "has_hand_semantics": has_hand,
        "aisle_numbers": aisle_numbers,
        "aisle_sides": aisle_sides,
        "path_specific": path_specific,   # e.g., 'Path (Office)'
        "cross_pairs": cross_pairs,       # e.g., '1-2'
    }


def resolve_display_label(col_name: str, code_map: Dict[str, str]) -> str:
    """
    Purpose:
        Resolve a DataFrame column name to its human label using the scheme code map if
        the column name is a code (e.g., 'CL065') or in the form 'CL065|Upwards'.

    Args:
        col_name (str): Column name in the CSV.
        code_map (Dict[str, str]): Mapping from code -> label.

    Returns:
        str: Human-readable label for this column or the original name.
    """
    # Handle "CL065|Upwards" by splitting; prefer code_map when possible.
    if "|" in col_name:
        code, label = col_name.split("|", 1)
        code = code.strip()
        label = label.strip()
        return code_map.get(code, label)
    # Raw code like "CL065"
    if col_name in code_map:
        return code_maps[col_name] if isinstance(code_maps := code_map, dict) and col_name in code_maps else code_map[col_name]
    return col_name


def find_columns_for_labels(df: pd.DataFrame,
                            desired_labels: List[str],
                            code_map: Dict[str, str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Purpose:
        From a list of target labels (from scheme), find which df columns correspond to them.
        A df column is considered a match if either:
        - the column name equals the label; or
        - the column name is a scheme code or 'code|label' whose mapped label equals the target label.

    Args:
        df (pd.DataFrame): Table whose columns we search (non-timestamp part typically).
        desired_labels (List[str]): Human labels from the scheme.
        code_map (Dict[str, str]): Mapping from code -> label.

    Returns:
        (matched_columns, display_map)
        matched_columns: List[str] column names found in df.
        display_map: Dict[str,str] mapping df column -> human label to display.
    """
    matched = []
    display_map = {}
    desired_set = set(desired_labels)

    for col in df.columns:
        label = resolve_display_label(col, code_map)
        if label in desired_set:
            matched.append(col)
            display_map[col] = label
    return matched, display_map


def join_active_binary_columns(df: pd.DataFrame,
                               columns: List[str],
                               display_map: Dict[str, str] = None) -> pd.Series:
    """
    Purpose:
        Build a string label per row by joining names of active (==1) binary columns.
        Column-wise logic (iterate columns; vectorize per column). If display_map is given,
        the output strings use display_map[col] instead of raw column names.

    Args:
        df (pd.DataFrame): Table containing binary columns.
        columns (List[str]): Column names to consider.
        display_map (Dict[str, str], optional): {df_column -> human_label}.

    Returns:
        pd.Series: Label strings like "Standing, Scan" or "-" if none active.
    """
    if not columns:
        return pd.Series(["-"] * len(df), index=df.index, dtype=object)

    out = np.full(len(df), "", dtype=object)
    disp = (display_map or {})

    for c in columns:
        if c not in df.columns:
            continue
        name = disp.get(c, resolve_display_label(c, {}))
        col = df[c].to_numpy()
        mask = (col == 1)
        need_sep = mask & (out != "")
        out[need_sep] = out[need_sep] + ", " + name
        out[mask & ~need_sep] = name

    out[out == ""] = "-"
    return pd.Series(out, index=df.index)


def choose_single_from_binary(df: pd.DataFrame,
                              columns: List[str],
                              display_map: Dict[str, str] = None) -> pd.Series:
    """
    Purpose:
        Choose a single active label per row (first matching column order wins).
        If none active, "-". Uses display_map for human-readable labels if provided.

    Args:
        df (pd.DataFrame): DataFrame with candidate binary columns.
        columns (List[str]): Ordered label columns (priority = order).
        display_map (Dict[str,str], optional): {df_column -> human_label}.

    Returns:
        pd.Series: The chosen label (human-readable) or "-".
    """
    n = len(df)
    out = np.full(n, "-", dtype=object)
    unset = np.ones(n, dtype=bool)
    disp = (display_map or {})

    for c in columns:
        if c not in df.columns:
            continue
        col = df[c].to_numpy()
        mask = (col == 1) & unset
        if not mask.any():
            continue
        label = disp.get(c, resolve_display_label(c, {}))
        out[mask] = label
        unset &= ~mask
        if not unset.any():
            break

    return pd.Series(out, index=df.index)


def build_aisle_combo(df: pd.DataFrame,
                      number_cols: List[str],
                      side_cols: List[str],
                      display_map: Dict[str, str] = None) -> pd.Series:
    """
    Purpose:
        Compose Aisle Path sub-locations as Number×Side (e.g., "1 front", "2 back", multiple joined).

    Args:
        df (pd.DataFrame): DataFrame containing number/side binary columns.
        number_cols (List[str]): df columns that correspond to numbers ("1","2"...).
        side_cols (List[str]): df columns that correspond to "Front"/"Back".
        display_map (Dict[str,str], optional): Used to get human labels for the numbers/sides.

    Returns:
        pd.Series: "-", "1 front", or "1 front, 2 back".
    """
    disp = (display_map or {})
    if not number_cols and not side_cols:
        return pd.Series(["-"] * len(df), index=df.index)

    n = len(df)
    out = np.full(n, "", dtype=object)

    nums = [(c, (df[c].to_numpy() == 1)) for c in number_cols if c in df.columns]
    sides = [(c, (df[c].to_numpy() == 1)) for c in side_cols if c in df.columns]

    for num_name, num_mask in nums:
        num_disp = disp.get(num_name, resolve_display_label(num_name, {}))
        for side_name, side_mask in sides:
            side_disp = disp.get(side_name, resolve_display_label(side_name, {})).lower()
            pair_mask = num_mask & side_mask
            if pair_mask.any():
                pair = f"{num_disp} {side_disp}"
                need_sep = pair_mask & (out != "")
                out[need_sep] = out[need_sep] + ", " + pair
                out[pair_mask & ~need_sep] = pair

    # If numbers exist but no sides are active, still allow lone numbers
    if sides == []:
        for num_name, num_mask in nums:
            num_disp = disp.get(num_name, resolve_display_label(num_name, {}))
            need_sep = num_mask & (out != "")
            out[need_sep] = out[need_sep] + ", " + num_disp
            out[num_mask & ~need_sep] = num_disp

    out[out == ""] = "-"
    return pd.Series(out, index=df.index)

def compose_location_labels(df: pd.DataFrame,
                            scheme_info: Dict[str, Any],
                            code_map: Dict[str, str]) -> Tuple[pd.Series, pd.Series]:
    """
    Purpose:
        Produce Location Main/Sub following the scheme hierarchy:
          - Main (single-choice): one of the main classes (Office, Cart Area, ..., Path, Cross Aisle Path, Aisle Path, ...)
          - Sub (gated by Main):
              * if Main == 'Path'             -> join active Path(Area) variants ('Path (Office)', ...)
              * if Main == 'Cross Aisle Path' -> join active cross pairs ('1-2', '2-3', ...)
              * if Main == 'Aisle Path'       -> compose number×side ('1 back, 2 front')
              * else                          -> '-'

    Args:
        df (pd.DataFrame): Non-timestamp columns for this location annotation (binary or text-like).
        scheme_info (Dict[str,Any]): Output of parse_scheme() with groups and semantic hints.
        code_map (Dict[str,str]): Mapping code->label for resolving df column names.

    Returns:
        (main_series, sub_series): pd.Series with human-readable Main and Sub.
    """
    groups = scheme_info["groups"]

    # ---- MAIN: single-choice among canonical main classes present in the scheme
    main_candidates = {
        "Office", "Cart Area", "Cardboard Box Area", "Base",
        "Packing/Sorting Area", "Issuing/Receiving Area",
        "Path", "Cross Aisle Path", "Aisle Path",
        "Another Location", "Location Unknown",
    }
    main_cols, main_disp = [], {}
    for g in groups:
        labels = [lbl for lbl in g["label_names"] if lbl in main_candidates]
        if not labels:
            continue
        c, d = find_columns_for_labels(df, labels, code_map)
        main_cols += c
        main_disp.update(d)

    main_series = choose_single_from_binary(df, main_cols, main_disp) if main_cols else pd.Series(["-"] * len(df), index=df.index)

    # ---- Build all Sub candidates (vectorized), but don't assign yet
    # A) Path(Area) variants
    path_labels = sorted(list(scheme_info.get("path_specific", []))) if scheme_info.get("path_specific") else []
    ps_cols, ps_disp = find_columns_for_labels(df, path_labels, code_map)
    sub_patharea = join_active_binary_columns(df, ps_cols, ps_disp) if ps_cols else pd.Series(["-"] * len(df), index=df.index)

    # B) Cross-aisle pairs
    cross_labels = sorted(list(scheme_info.get("cross_pairs", []))) if scheme_info.get("cross_pairs") else []
    cp_cols, cp_disp = find_columns_for_labels(df, cross_labels, code_map)
    sub_cross = join_active_binary_columns(df, cp_cols, cp_disp) if cp_cols else pd.Series(["-"] * len(df), index=df.index)

    # C) Aisle number × side
    number_labels = sorted(list(scheme_info.get("aisle_numbers", [])), key=lambda x: int(x)) if scheme_info.get("aisle_numbers") else []
    side_labels   = list(scheme_info.get("aisle_sides", [])) if scheme_info.get("aisle_sides") else []
    num_cols, num_disp   = find_columns_for_labels(df, number_labels, code_map)
    side_cols, side_disp = find_columns_for_labels(df, side_labels,   code_map)
    disp_total = {**num_disp, **side_disp}
    sub_aisle = build_aisle_combo(df, num_cols, side_cols, disp_total) if (num_cols or side_cols) else pd.Series(["-"] * len(df), index=df.index)

    # ---- Gate Sub strictly by the chosen Main value (scheme-driven)
    main_str = main_series.astype(str)
    sub_series = pd.Series(["-"] * len(df), index=df.index, dtype=object)

    # Only the matching branch contributes; others stay '-'
    sub_series = pd.Series(
        np.where(main_str == "Path", sub_patharea, sub_series), index=df.index, dtype=object
    )
    sub_series = pd.Series(
        np.where(main_str == "Cross Aisle Path", sub_cross, sub_series), index=df.index, dtype=object
    )
    sub_series = pd.Series(
        np.where(main_str == "Aisle Path", sub_aisle, sub_series), index=df.index, dtype=object
    )

    return main_series.fillna("-"), sub_series.fillna("-")



def compose_hand_labels(df: pd.DataFrame,
                        scheme_info: Dict[str, Any],
                        code_map: Dict[str, str]) -> pd.Series:
    """
    Purpose:
        Build a single string label for hand annotations by concatenating major groups
        (Primary Position / Type of Movement / Object / Tool). Each sub-group is treated
        as single-choice by default (first active wins); if multiple are active, we join them.

    Args:
        df (pd.DataFrame): Non-timestamp columns for the hand annotation.
        scheme_info (Dict[str,Any]): Output of parse_scheme().
        code_map (Dict[str,str]): Mapping code->label (resolves df column names to human labels).

    Returns:
        pd.Series: "Primary Position / Type of Movement / Object / Tool"
    """
    def pick_groups(keywords: List[str]) -> List[Dict[str, Any]]:
        out = []
        for g in scheme_info["groups"]:
            name_low = g["name"].lower()
            if any(k in name_low for k in keywords):
                out.append(g)
        return out

    g_primary = pick_groups(["primary position"])
    g_movement = pick_groups(["type of movement"])
    g_object   = pick_groups(["object"])
    g_tool     = pick_groups(["tool"])

    def build_for(groups_list: List[Dict[str, Any]]) -> pd.Series:
        if not groups_list:
            return pd.Series(["-"] * len(df), index=df.index)
        # union of labels that exist as columns (match on code or label)
        cols, disp = [], {}
        for g in groups_list:
            c, d = find_columns_for_labels(df, g["label_names"], code_map)
            cols += c
            disp.update(d)
        if not cols:
            return pd.Series(["-"] * len(df), index=df.index)

        # Prefer single-choice; if multiple active in a row, join them
        s_single = choose_single_from_binary(df, cols, disp)
        multi_mask = (df[cols] == 1).sum(axis=1) > 1
        s_join = join_active_binary_columns(df, cols, disp)
        return pd.Series(np.where(multi_mask, s_join, s_single), index=df.index)

    part_primary   = build_for(g_primary)
    part_movement  = build_for(g_movement)
    part_object    = build_for(g_object)
    part_tool      = build_for(g_tool)

    out = (part_primary.fillna("-") + " / " +
           part_movement.fillna("-") + " / " +
           part_object.fillna("-") + " / " +
           part_tool.fillna("-"))
    return out.replace(r"^\s*/\s*|\s*/\s*$", "-", regex=True)


def vector_labels_from_groups(df: pd.DataFrame, scheme_info: Dict[str, Any], code_map: Dict[str, str]) -> pd.Series:
    """
    Purpose:
        Generic extractor when no special semantics (Location/Hand) apply.
        If a scheme has multiple named groups, we build a compact label by
        concatenating each group's (single or joined) label with " / ".

    Args:
        df (pd.DataFrame): Non-timestamp columns for this annotation.
        scheme_info (Dict[str,Any]): Output of parse_scheme().
        code_map (Dict[str,str]): Mapping code->label (resolves df column names to human labels).

    Returns:
        pd.Series: Compact combined label across groups, or joined-active if no groups found.
    """
    groups = scheme_info["groups"]
    if not groups:
        # fallback: join everything that looks binary
        bin_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not bin_cols:
            last = df.columns[-1] if len(df.columns) else None
            return df[last].astype(str) if last else pd.Series(["-"] * len(df), index=df.index)
        return join_active_binary_columns(df, bin_cols)

    parts = []
    for g in groups:
        cols, disp = find_columns_for_labels(df, g["label_names"], code_map)
        if not cols:
            parts.append(pd.Series(["-"] * len(df), index=df.index))
            continue
        s_single = choose_single_from_binary(df, cols, disp)
        multi_mask = (df[cols] == 1).sum(axis=1) > 1
        s_join = join_active_binary_columns(df, cols, disp)
        s = pd.Series(np.where(multi_mask, s_join, s_single), index=df.index)
        parts.append(s)

    # stitch group parts with " / "
    out = parts[0]
    for s in parts[1:]:
        out = (out.fillna("-") + " / " + s.fillna("-"))
    return out

# -------- Master builder (replacement for extract_label_column) --------

def build_labels_from_annotations(df: pd.DataFrame,
                                  desired_name: str,
                                  scheme_key: str,
                                  code_maps: Dict[str, Dict[str, str]],
                                  raw_scheme: Any) -> pd.Series:
    """
    Purpose:
        Fast, scheme-driven label builder to replace 'extract_label_column'.
        Uses the scheme to understand groups (hands, location, etc.), and composes
        a compact, human-readable label string per row (column-wise, vectorized).

    Args:
        df (pd.DataFrame): Full category table; first column is timestamp; remaining columns are labels/binaries.
        desired_name (str): Name to assign to the resulting label series (column in merged table).
        scheme_key (str): Like "CC11_Location - Human", "CC05_Sub-Activity - Right Hand", etc.
        code_maps (Dict[str, Dict[str, str]]): scheme -> { "CLxxx": "Label" } for mapping codes.
        raw_scheme (Any): Parsed JSON structure for this scheme.

    Returns:
        pd.Series: Label strings per row, named 'desired_name'.
    """
    if df.empty:
        return pd.Series(dtype=object, name=desired_name)

    label_df = df.copy()

    # Fast path: already a single text label column
    if label_df.shape[1] == 1 and label_df.dtypes.iloc[0] == object:
        s = label_df.iloc[:, 0].astype(str).fillna("-")
        s.name = desired_name
        return s

    scheme_info = parse_scheme(raw_scheme or [])
    code_map = code_maps.get(scheme_key, {})

    if scheme_info["has_location_semantics"]:
        main_s, sub_s = compose_location_labels(label_df, scheme_info, code_map)
        s = (main_s.fillna("-") + " / " + sub_s.fillna("-"))
    elif scheme_info["has_hand_semantics"]:
        s = compose_hand_labels(label_df, scheme_info, code_map)
    else:
        s = vector_labels_from_groups(label_df, scheme_info, code_map)

    s.name = desired_name
    return s

# ----------------------------- Concatenate (no timestamp) -------------------------------

def concat_labels_by_index(named_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Purpose:
        Concatenate per-category label columns by **row index** (timestamps are not used).
        Assumes synchronized streams (same sampling/order). If lengths differ, we keep the
        intersection of indices (inner join) to avoid misalignment.

    Args:
        named_frames (Dict[str, pd.DataFrame]):
            {'Main Activity': df_main_use, 'Location – Human': df_loc_h_use, ...}
            Each df must contain exactly one column (its label).

    Returns:
        pd.DataFrame:
            A table of shape [N_common_rows, n_categories] with friendly column names.
    """
    if not named_frames:
        return pd.DataFrame()

    # Ensure each df has exactly one column; if not, pick the last column as the label.
    cleaned = []
    for friendly, df in named_frames.items():
        if df is None or df.empty:
            continue
        cols = list(df.columns)
        if len(cols) != 1:
            df = df[[cols[-1]]].copy()
        df.columns = [friendly]
        cleaned.append(df)

    if not cleaned:
        return pd.DataFrame()

    merged = pd.concat(cleaned, axis=1, join="inner")
    return merged

# ----------------------------- Filtering -------------------------------

def filter_unknown_and_other(df: pd.DataFrame,
                             remove_unknown: bool,
                             remove_other: bool,
                             scheme_name_map: Dict[str, str],
                             unknown_sets: Dict[str, set],
                             other_sets: Dict[str, set]) -> pd.DataFrame:
    """
    Purpose:
        Remove rows containing 'Unknown' and/or 'Another/Other' labels, using the scheme sets.

    Args:
        df (pd.DataFrame): Table to filter (label-only columns).
        remove_unknown (bool): If True, drop rows that contain any 'Unknown' label.
        remove_other (bool): If True, drop rows that contain any 'Another/Other' label.
        scheme_name_map (Dict[str, str]): {column_display_name -> scheme_base_name}.
        unknown_sets (Dict[str, set]): {scheme_base_name -> set(str_labels_marked_unknown)}.
        other_sets (Dict[str, set]): {scheme_base_name -> set(str_labels_marked_other)}.

    Returns:
        pd.DataFrame: Filtered table (index reset).
    """
    if not remove_unknown and not remove_other:
        return df

    keep_mask = pd.Series(True, index=df.index)
    for col in df.columns:
        scheme_key = scheme_name_map.get(col, "")
        col_vals = df[col].astype(str)

        if remove_unknown and scheme_key in unknown_sets and unknown_sets[scheme_key]:
            keep_mask &= ~col_vals.isin(unknown_sets[scheme_key])

        if remove_other and scheme_key in other_sets and other_sets[scheme_key]:
            keep_mask &= ~col_vals.isin(other_sets[scheme_key])

        # Textual fallback
        if remove_unknown:
            keep_mask &= ~col_vals.str.lower().str.contains("unknown")
        if remove_other:
            keep_mask &= ~(col_vals.str.lower().str.contains("another") |
                           col_vals.str.lower().str.contains("other"))

    return df.loc[keep_mask].reset_index(drop=True)

# ----------------------------- Location (Main/Sub) expansion -------------------------------

def split_location_value(value: str) -> Tuple[str, str]:
    """
    Purpose:
        Split a location label into (Main, Sub) parts using common separators.

    Args:
        value (str): Original location label (e.g., "Office / -", "Aisle-3 - Slot-12", "Office").

    Returns:
        Tuple[str, str]: (main_location, sub_location). Missing parts are returned as "-".
    """
    if not value or value == "-":
        return "-", "-"
    s = str(value).strip()
    # Try a set of separators in order of likelihood
    seps = [" / ", "/", " - ", " – ", "—", "-", ",", "|"]
    for sep in seps:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p is not None]
            if len(parts) == 1:
                return parts[0] if parts[0] else "-", "-"
            if len(parts) >= 2:
                main = parts[0] if parts[0] else "-"
                sub  = parts[1] if parts[1] else "-"
                return main, sub
    # No separator found → treat entire as Main
    return s, "-"


def expand_location_columns(df: pd.DataFrame,
                            col_label: str,
                            main_col_label: str,
                            sub_col_label: str) -> pd.DataFrame:
    """
    Purpose:
        From a single location column (e.g., 'Location – Human'), derive two columns:
        '{col_label} (Main)' and '{col_label} (Sub)'.

    Args:
        df (pd.DataFrame): Merged table containing 'col_label'.
        col_label (str): Source column name (e.g., 'Location – Human').
        main_col_label (str): New name for main location column.
        sub_col_label (str): New name for sub location column.

    Returns:
        pd.DataFrame: Same df with the two new columns appended (source column is retained).
    """
    if col_label not in df.columns:
        df[main_col_label] = "-"
        df[sub_col_label] = "-"
        return df

    mains, subs = [], []
    for v in df[col_label].astype(str).fillna("-").tolist():
        m, s = split_location_value(v)
        mains.append(m if m else "-")
        subs.append(s if s else "-")

    df[main_col_label] = pd.Series(mains, index=df.index)
    df[sub_col_label] = pd.Series(subs, index=df.index)
    return df

# ----------------------------- Hand (Primary Position / Movement / Object / Tool) expansion -------------------------------

def extract_hand_group_series(
    df: pd.DataFrame,
    group_label_names: List[str],
    code_map: Dict[str, str],
) -> pd.Series:
    """
    Purpose:
        Build a per-row human-readable label for ONE hand sub-group (e.g., Primary Position),
        using column-wise vectorization over likely one-hot columns.

    Args:
        df: Non-timestamp annotation matrix for this hand category.
        group_label_names: The human labels that belong to this group, from the scheme.
        code_map: code -> label mapping for this scheme.

    Returns:
        pd.Series with a single label per row:
          - If exactly one column in the group is active, returns that label.
          - If multiple are active, joins them with ', '.
          - If none active, returns '-'.
    """
    cols, disp = find_columns_for_labels(df, group_label_names, code_map)
    if not cols:
        return pd.Series(["-"] * len(df), index=df.index, dtype=object)

    # Choose single if possible; otherwise join actives.
    s_single = choose_single_from_binary(df, cols, disp)  # first active wins
    multi_mask = (df[cols] == 1).sum(axis=1) > 1
    s_join = join_active_binary_columns(df, cols, disp)
    return pd.Series(np.where(multi_mask, s_join, s_single), index=df.index, dtype=object)


def expand_hand_columns_from_scheme(
    df_raw: pd.DataFrame,
    base_display_name: str,
    scheme_key: str,
    code_maps: Dict[str, Dict[str, str]],
    raw_scheme: Any,
) -> Dict[str, pd.Series]:
    """
    Purpose:
        Split a hand annotation ('Sub-Activity – Left Hand' / 'Sub-Activity – Right Hand') into
        four columns using the scheme groups:
            (Primary Position), (Type of Movement), (Object), (Tool)

    Args:
        df_raw: Full raw hand DataFrame for a subject (timestamp in col0, features after).
        base_display_name: Friendly name, e.g. 'Sub-Activity – Left Hand' or 'Sub-Activity – Right Hand'.
        scheme_key: e.g. 'CC04_Sub-Activity - Left Hand' or 'CC05_Sub-Activity - Right Hand'.
        code_maps: scheme -> { 'CLxxx': 'Label' } lookup.
        raw_scheme: Parsed JSON for this scheme (list of [GroupName, [ 'CL|Label', ... ]]).

    Returns:
        Dict[str, pd.Series]:
            {
              '<base> (Primary Position)': Series,
              '<base> (Type of Movement)': Series,
              '<base> (Object)': Series,
              '<base> (Tool)': Series
            }
            Missing groups (if any) are returned as '-' series.
    """
    if df_raw is None or df_raw.empty:
        return {}

    # Drop the time-like first column if present; we only want annotation columns.
    label_df = df_raw.iloc[:, 1:].copy() if df_raw.shape[1] > 1 else df_raw.copy()

    scheme_info = parse_scheme(raw_scheme or [])
    code_map = code_maps.get(scheme_key, {})

    # Find the four groups by fuzzy name match (scheme text drives this; no hardcoding of codes).
    # The left-hand scheme includes the exact group names shown in your JSON (same for the right hand).
    group_map = {"Primary Position": None, "Type of Movement": None, "Object": None, "Tool": None}
    for g in scheme_info["groups"]:
        name_low = g["name"].strip().lower()
        if "primary position" in name_low:
            group_map["Primary Position"] = g["label_names"]
        elif "type of movement" in name_low:
            group_map["Type of Movement"] = g["label_names"]
        elif name_low == "object":
            group_map["Object"] = g["label_names"]
        elif name_low == "tool":
            group_map["Tool"] = g["label_names"]

    out = {}
    for group_name, labels in group_map.items():
        if labels:
            s = extract_hand_group_series(label_df, labels, code_map)
        else:
            s = pd.Series(["-"] * len(label_df), index=label_df.index, dtype=object)
        out[f"{base_display_name} ({group_name})"] = s
    return out

# ----------------------------- Combos & Mappings -------------------------------

def add_combination_columns(df: pd.DataFrame,
                            input_cols: List[str] = None,
                            output_cols: List[str] = None) -> pd.DataFrame:
    """
    Purpose:
        Create 'input' and/or 'output' combination columns in the merged table using
        the exact column lists provided (order preserved), and print which columns
        were combined for each.

    Args:
        df (pd.DataFrame): Merged table (label-only columns).
        input_cols (List[str], optional): Columns to combine for 'input'. If None/empty, 'input' not added.
        output_cols (List[str], optional): Columns to combine for 'output'. If None/empty, 'output' not added.

    Returns:
        pd.DataFrame: Same as input with optional 'input' and/or 'output' columns appended.
    """
    if df.empty:
        return df

    # --- INPUT COMBINATION ---
    if input_cols:
        cols = [c for c in input_cols if c in df.columns]
        if cols:
            df["input"] = df[cols].fillna("-").agg("/".join, axis=1)
            print("\n📦 Created INPUT combination using:")
            for c in cols:
                print(f"   - {c}")
        else:
            df["input"] = "-"
            print("\n⚠ No valid columns found for INPUT combination.")

    # --- OUTPUT COMBINATION ---
    if output_cols:
        cols = [c for c in output_cols if c in df.columns]
        if cols:
            df["output"] = df[cols].fillna("-").agg("/".join, axis=1)
            print("\n🧩 Created OUTPUT combination using:")
            for c in cols:
                print(f"   - {c}")
        else:
            df["output"] = "-"
            print("\n⚠ No valid columns found for OUTPUT combination.")

    return df


def friendly_to_scheme_key(friendly: str) -> str:
    """
    Purpose:
        Convert a displayed category name to the likely scheme base name.

    Args:
        friendly (str): e.g., "Main Activity", "Location – Human", "Mid-Level Process".

    Returns:
        str: Likely scheme key used in scheme filenames.
    """
    manual = {
        "Main Activity": "CC01_Main Activity",
        "Order": "CC06_Order",
        "Information Technology": "CC07_Information Technology",
        "High-Level Process": "CC08_High-Level Process",
        "Mid-Level Process": "CC09_Mid-Level Process",
        "Low-Level Process": "CC10_Low-Level Process",
        "Location – Human": "CC11_Location - Human",
        "Location - Human": "CC11_Location - Human",
        "Location – Cart": "CC12_Location - Cart",
        "Location - Cart": "CC12_Location - Cart",
        "Sub-Activity – Legs": "CC02_Sub-Activity - Legs",
        "Sub-Activity – Torso": "CC03_Sub-Activity - Torso",
        "Sub-Activity – Left Hand": "CC04_Sub-Activity - Left Hand",
        "Sub-Activity – Right Hand": "CC05_Sub-Activity - Right Hand",
    }
    return manual.get(friendly, friendly)

# ---------------------------- Main flow ------------------------------

def main():
    """
    Purpose:
        Run the interactive workflow:
        - Detect categories
        - Ask for selections and options
        - Load scheme maps (+ raw scheme)
        - For each subject: read selected categories, build label columns (scheme-aware),
          expand Location (Main/Sub) and Hand (Primary/Movement/Object/Tool) columns,
          **concat by index** (no timestamp), filter (Unknown/Another),
          ask which columns to combine (input/output), save CSV.

    Inputs:
        Reads from 'Annotation_Rostock/raw_data/'.
        Scheme files from 'Annotation_Rostock/raw_data/scheme/'.

    Outputs:
        Writes CSVs to 'Annotation_Rostock/data_preprocessed/SXX.csv'.
    """

    # Make sure raw_data/ exists (or unzip raw_data.zip if needed)
    ensure_raw_data_ready()

    # --- rest of your existing main() stays the same ---
    category_paths = detect_annotation_folders(RAW_DIR)
    if not category_paths:
        print(f"✖ No annotation folders found under: {RAW_DIR}")
        print("   Make sure 'raw_data/' exists or 'raw_data.zip' is placed next to this script.")
        sys.exit(1)

    ordered_categories = list(category_paths.keys())
    chosen_categories = choose_from_menu(ordered_categories)

    # User options
    remove_unknown = ask_yes_no("Do you want to remove 'Unknown' activities/processes/locations?")
    remove_other = ask_yes_no("Do you want to remove 'Another/Other' activities/processes/locations?")
    make_input_combo = ask_yes_no("Do you want to create INPUT combinations (activity + sub-activities + locations + IT + order)?")
    make_output_combo = ask_yes_no("Do you want to create OUTPUT combinations (High/Mid/Low-level processes)?")

    # If combos are requested, ask which of the *selected* categories to combine
    input_combo_cols: List[str] = []
    output_combo_cols: List[str] = []

    # Location part selections (affect INPUT column composition)
    use_location_human_parts: List[str] = []  # subset of ["Main", "Sub"]
    use_location_cart_parts: List[str] = []   # subset of ["Main", "Sub"]

    # Hand part selections (affect INPUT column composition)
    use_left_hand_parts: List[str] = []   # subset of ["Primary Position","Type of Movement","Object","Tool"]
    use_right_hand_parts: List[str] = []  # same

    if make_input_combo:
        candidates_in = [c for c in INPUT_FAMILY if c in chosen_categories]
        if not candidates_in:
            print("⚠ No suitable columns among your selection to build an INPUT combo; skipping.")
        else:
            input_combo_cols = choose_combo_columns(candidates_in, "INPUT combination")

            # Ask sub-part choices for Location
            if "Location – Human" in input_combo_cols:
                use_location_human_parts = choose_location_parts("Location – Human")
            if "Location – Cart" in input_combo_cols:
                use_location_cart_parts = choose_location_parts("Location – Cart")

            # Ask sub-part choices for Hands
            if "Sub-Activity – Left Hand" in input_combo_cols:
                use_left_hand_parts = choose_hand_parts("Sub-Activity – Left Hand")
            if "Sub-Activity – Right Hand" in input_combo_cols:
                use_right_hand_parts = choose_hand_parts("Sub-Activity – Right Hand")

    if make_output_combo:
        candidates_out = [c for c in OUTPUT_FAMILY if c in chosen_categories]
        if not candidates_out:
            print("⚠ No suitable columns among your selection to build an OUTPUT combo; skipping.")
        else:
            output_combo_cols = choose_combo_columns(candidates_out, "OUTPUT combination")

    # Load JSON scheme info (with raw)
    code_maps, unknown_sets, other_sets, raw_by_key = read_scheme_files_with_raw(SCHEME_DIR)

    # Prepare output
    os.makedirs(OUT_DIR, exist_ok=True)

    # Map display names -> scheme keys
    scheme_name_map = {disp: friendly_to_scheme_key(disp) for disp in chosen_categories}

    # Read all chosen categories once
    print("\nLoading category tables ...")
    per_category_subject: Dict[str, Dict[str, pd.DataFrame]] = {}
    for disp_name in chosen_categories:
        per_category_subject[disp_name] = read_subject_tables(category_paths[disp_name])

    # Process each subject
    print("Merging data subject-wise and saving outputs ...")
    for subj in SUBJECTS:
        per_name_df: Dict[str, pd.DataFrame] = {}

        # Build label columns and expand structured parts per chosen category
        for disp_name in chosen_categories:
            df_raw = per_category_subject.get(disp_name, {}).get(subj)
            if df_raw is None or df_raw.empty:
                continue

            scheme_key = scheme_name_map.get(disp_name, disp_name)
            raw_scheme = raw_by_key.get(scheme_key, [])

            # Base compact label (scheme-aware, column-wise)
            label_series = build_labels_from_annotations(df_raw, disp_name, scheme_key, code_maps, raw_scheme)
            per_name_df[disp_name] = pd.DataFrame({disp_name: label_series})

            # Always expand Hands into 4 parts if present (Primary/Movement/Object/Tool)
            if disp_name in ("Sub-Activity – Left Hand", "Sub-Activity – Right Hand"):
                hand_parts = expand_hand_columns_from_scheme(
                    df_raw=df_raw,
                    base_display_name=disp_name,
                    scheme_key=scheme_key,
                    code_maps=code_maps,
                    raw_scheme=raw_scheme,
                )
                for col_name, series in hand_parts.items():
                    per_name_df[col_name] = pd.DataFrame({col_name: series})

        if not per_name_df:
            continue

        # Concat by index (no timestamps)
        merged = concat_labels_by_index(per_name_df)

        # Filter unknown/other via scheme sets (plus textual fallback)
        merged = filter_unknown_and_other(
            merged,
            remove_unknown=remove_unknown,
            remove_other=remove_other,
            scheme_name_map=scheme_name_map,
            unknown_sets=unknown_sets,
            other_sets=other_sets
        )

        # Always expand Location columns (if present), regardless of combo choices
        if merged.shape[0] > 0:
            if "Location – Human" in merged.columns:
                merged = expand_location_columns(
                    merged,
                    "Location – Human",
                    "Location – Human (Main)",
                    "Location – Human (Sub)"
                )
            if "Location – Cart" in merged.columns:
                merged = expand_location_columns(
                    merged,
                    "Location – Cart",
                    "Location – Cart (Main)",
                    "Location – Cart (Sub)"
                )

        # Construct final input/output column lists according to user choices
        final_input_cols = None
        if input_combo_cols:
            final_input_cols = []
            for c in input_combo_cols:
                if c == "Location – Human" and use_location_human_parts:
                    if "Main" in use_location_human_parts:
                        final_input_cols.append("Location – Human (Main)")
                    if "Sub" in use_location_human_parts:
                        final_input_cols.append("Location – Human (Sub)")

                elif c == "Location – Cart" and use_location_cart_parts:
                    if "Main" in use_location_cart_parts:
                        final_input_cols.append("Location – Cart (Main)")
                    if "Sub" in use_location_cart_parts:
                        final_input_cols.append("Location – Cart (Sub)")

                elif c == "Sub-Activity – Left Hand" and use_left_hand_parts:
                    part_map = {
                        "Primary Position": f"{c} (Primary Position)",
                        "Type of Movement": f"{c} (Type of Movement)",
                        "Object":           f"{c} (Object)",
                        "Tool":             f"{c} (Tool)",
                    }
                    for p in use_left_hand_parts:
                        colname = part_map.get(p)
                        if colname:
                            final_input_cols.append(colname)

                elif c == "Sub-Activity – Right Hand" and use_right_hand_parts:
                    part_map = {
                        "Primary Position": f"{c} (Primary Position)",
                        "Type of Movement": f"{c} (Type of Movement)",
                        "Object":           f"{c} (Object)",
                        "Tool":             f"{c} (Tool)",
                    }
                    for p in use_right_hand_parts:
                        colname = part_map.get(p)
                        if colname:
                            final_input_cols.append(colname)

                else:
                    final_input_cols.append(c)

        final_output_cols = output_combo_cols if output_combo_cols else None

        # Add combo columns (using exactly what the user selected)
        merged = add_combination_columns(
            merged,
            input_cols=final_input_cols,
            output_cols=final_output_cols
        )

        # Fill missing with "-"
        merged = merged.fillna("-")

        # Save
        out_path = os.path.join(OUT_DIR, f"{subj}.csv")
        merged.to_csv(out_path, index=False)
        print(f"✔ Saved {out_path} ({len(merged)} rows)")

    print("\nℹ️ Files are saved as S01 ... S18 in:")
    print(f"   {OUT_DIR}")
    print("   Where a category has no data, '-' is used.")
    print("   Example: Office has no sub-location, so '-' will appear when Office is the main location.")
    print("\nDone.")


if __name__ == "__main__":
    main()
