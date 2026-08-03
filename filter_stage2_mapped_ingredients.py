"""
Stage 2: loc tiep tren final_recipes_filtered.csv (ket qua cua filter_final_recipes_30k.py)
theo thu tu nguoi dung de xuat, dua tren mapped_ingredients (nguyen lieu da chuan hoa)
thay vi ingredients tho:

  B1: mapped_ingredients >= 3 nguyen lieu khac nhau, khong toan token rac (water/salt/pepper)
  B2: directions du dai (>= 80 ky tu)
  B3: exact duplicate theo title (chuan hoa) + sorted(set(mapped_ingredient_names))
  B4: mo rong blacklist khong-phai-am-thuc (glue, fertilizer, bird seed, bait, compost, face mask, bath bomb, ...)
  B5: recipe qua chung chung (lunch/dinner/breakfast/menu/ideas/collection + qua nhieu nguyen lieu)
  B6: near-duplicate fuzzy CHAT hon (title_sim > 95 AND mapped_ingredient jaccard > 0.9) -> tu dong
      giu 1 ban dai dien (khong can review vi nguong rat chat)

Output:
  - final_recipes_filtered_v2.csv
  - filter_log_v2.csv (id, title, status, reason)
"""

import json
import re
import time
from collections import defaultdict
from itertools import combinations

import pandas as pd
import requests
from rapidfuzz import fuzz
from tqdm import tqdm

INPUT_CSV = "final_recipes_filtered.csv"
OUTPUT_CSV = "final_recipes_filtered_v2.csv"
LOG_CSV = "filter_log_v2.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:7b-instruct"
MIN_RAW_INGREDIENTS = 3
LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "is_ingredients_adequate": {"type": "boolean"},
        "is_directions_clear": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_ingredients_adequate", "is_directions_clear", "reason"],
}
LLM_PROMPT_TEMPLATE = """This recipe's ingredient list may be short due to a data-mapping error, not necessarily a bad recipe. Judge it on its own merits.

is_ingredients_adequate: true if the ingredients listed are plausibly enough to make the dish named in the title (false only if clearly missing essential components or is just 1-2 generic items with no real dish description).
is_directions_clear: true if the directions give clear, specific, actionable steps to prepare the dish (false if vague/empty, e.g. just "mix and serve" with no real detail, or missing).

title="{title}"
ingredients={ingredients}
directions={directions}
"""

MIN_MAPPED_INGREDIENTS = 3
MIN_NON_JUNK_INGREDIENTS = 2
MIN_DIRECTIONS_CHARS = 80

JUNK_KEYWORD_PREFIXES = ["water", "salt", "spices, pepper", "spices, salt", "ice"]

NONFOOD_PATTERNS = [
    r"\bsoap\b", r"\bshampoo\b", r"\bdetergent\b", r"\bcleaner\b", r"\bcleaning\b",
    r"\bslime\b", r"\bglue\b", r"\bfertilizer\b", r"\bweed\s+killer\b",
    r"\bbird\s+seed\b", r"\bface\s+mask\b",
    r"\bbath\s+bomb\b", r"\bbug\s+spray\b", r"\binsect\s+repellent\b",
    r"\bsunscreen\b", r"\bcandle\b", r"\(not\s+edible\)", r"\bnot\s+edible\b",
    r"\blaundry\b", r"\bplay\s*dough\b", r"\bpotpourri\b",
]
# Luu y: bo "bait" va "compost" khoi blacklist - test thuc te cho thay chung bat nham
# ten mon that ("Compost Cookies", "Blueberry Boy Bait", "Man Bait" la ten choi chu cho banh).
NONFOOD_RE = re.compile("|".join(f"(?:{p})" for p in NONFOOD_PATTERNS), re.IGNORECASE)

# Luu y: bo "lunch/dinner/breakfast" khoi day - trong tieng Anh day thuong LA mot phan
# ten mon cu the (vd "Breakfast Burrito", "Dinner Rolls"), khong phai dau hieu gom nhieu mon.
# Chi giu lai menu/ideas/collection la nhung tu hiem va it false positive hon.
GENERIC_MEAL_PATTERNS = [
    r"\bmenu\b", r"\bideas\b", r"\bcollection\b",
]
GENERIC_MEAL_RE = re.compile("|".join(f"(?:{p})" for p in GENERIC_MEAL_PATTERNS), re.IGNORECASE)
GENERIC_MAX_INGREDIENTS = 12  # tren nguong nay + tu khoa generic => coi la "gom nhieu mon"

TITLE_PREFIXES = [
    "easy", "best", "the best", "ultimate", "quick", "quick and easy",
    "classic", "simple", "perfect", "homemade", "old fashioned",
    "old-fashioned", "traditional", "authentic", "famous", "super",
    "delicious", "amazing", "yummy", "favorite", "favourite",
    "award winning", "world's best", "worlds best", "no-fail", "nofail",
    "foolproof", "grandma's", "grandmas", "mom's", "moms", "my",
]
TITLE_SUFFIXES = ["recipe", "recipes"]

FUZZY_TITLE_THRESHOLD = 95
FUZZY_JACCARD_THRESHOLD = 0.9


def normalize_title(title):
    t = str(title).strip().lower()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    changed = True
    while changed:
        changed = False
        for suf in TITLE_SUFFIXES:
            if t.endswith(" " + suf):
                t = t[: -(len(suf) + 1)].strip()
                changed = True
        for pre in TITLE_PREFIXES:
            if t.startswith(pre + " "):
                t = t[len(pre) + 1:].strip()
                changed = True
    return t


def parse_mapped_names(raw):
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(items, list):
        return []
    names = []
    for e in items:
        if isinstance(e, dict) and e.get("mapped_name"):
            names.append(str(e["mapped_name"]).strip())
    return names


def is_junk_name(name):
    lname = name.lower()
    return any(lname.startswith(p) for p in JUNK_KEYWORD_PREFIXES)


def parse_json_list(raw):
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return items if isinstance(items, list) else []


def llm_check_ingredients_and_directions(title, ingredients, directions, max_retries=3):
    prompt = LLM_PROMPT_TEMPLATE.format(
        title=title, ingredients=json.dumps(ingredients), directions=json.dumps(directions))
    for attempt in range(max_retries):
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": LLM_MODEL, "prompt": prompt, "stream": False,
                "format": LLM_SCHEMA, "options": {"temperature": 0},
            }, timeout=60)
            r.raise_for_status()
            data = json.loads(r.json()["response"])
            return bool(data["is_ingredients_adequate"]), bool(data["is_directions_clear"]), data.get("reason", "")
        except Exception as e:
            print(f"  [warn] LLM check failed (attempt {attempt+1}): {e}", flush=True)
            time.sleep(2)
    # neu LLM loi lien tuc, mac dinh giu lai (tranh xoa nham do loi ky thuat)
    return True, True, "error_defaulted_keep"


def main():
    df = pd.read_csv(INPUT_CSV)
    df["id"] = df["id"].astype(str)
    log = {rid: {"title": title, "status": "kept", "reason": ""}
           for rid, title in zip(df["id"], df["title"])}

    def mark_removed(ids, reason):
        for rid in ids:
            if log[rid]["status"] == "kept":
                log[rid]["status"] = "removed"
                log[rid]["reason"] = reason

    total_start = len(df)
    df["_mapped_names"] = df["mapped_ingredients"].apply(parse_mapped_names)
    df["_raw_ingredients"] = df["ingredients"].apply(parse_json_list)

    # ---------- B1: mapped_ingredients >= 3, khong toan token rac ----------
    # An toan: chi tu dong loai khi CA mapped_ingredients LAN ingredients (tho) deu it
    # (tranh xoa nham cac recipe bi loi anh xa mapped_ingredients nhung ingredients tho van du).
    # Cac case mapped it nhung ingredients tho du (nghi ngo loi anh xa) se duoc LLM phan xu,
    # dong thoi LLM cung cham luon do ro rang cua directions cho nhung case nay.
    def unique_non_junk_count(names):
        uniq = set(n.lower() for n in names)
        non_junk = [n for n in uniq if not is_junk_name(n)]
        return len(uniq), len(non_junk)

    counts = df["_mapped_names"].apply(unique_non_junk_count)
    df["_n_unique"] = counts.apply(lambda t: t[0])
    df["_n_non_junk"] = counts.apply(lambda t: t[1])
    df["_n_raw"] = df["_raw_ingredients"].apply(len)

    mapped_low = (df["_n_unique"] < MIN_MAPPED_INGREDIENTS) | (df["_n_non_junk"] < MIN_NON_JUNK_INGREDIENTS)
    raw_low = df["_n_raw"] < MIN_RAW_INGREDIENTS

    b1_clear_remove_mask = mapped_low & raw_low
    mark_removed(df.loc[b1_clear_remove_mask, "id"], "b1_too_few_ingredients_both")

    b1_ambiguous_mask = mapped_low & ~raw_low
    ambiguous_ids = df.loc[b1_ambiguous_mask, "id"].tolist()
    print(f"[B1] loai {b1_clear_remove_mask.sum()} recipe qua it nguyen lieu (ca mapped va tho deu it)")
    print(f"[B1] {len(ambiguous_ids)} recipe mapped_ingredients it nhung ingredients tho du "
          f"(nghi ngo loi anh xa) -> hoi LLM", flush=True)

    llm_keep_ids, llm_remove_ids = set(), set()
    for i, rid in enumerate(ambiguous_ids):
        row = df.loc[df["id"] == rid].iloc[0]
        ing_ok, dir_ok, reason = llm_check_ingredients_and_directions(
            row["title"], row["_raw_ingredients"][:12], parse_json_list(row["directions"])[:6])
        if ing_ok and dir_ok:
            llm_keep_ids.add(rid)
        else:
            llm_remove_ids.add(rid)
            which = [] if ing_ok else ["ingredients"]
            which += [] if dir_ok else ["directions"]
            log[rid]["status"] = "removed"
            log[rid]["reason"] = f"b1_llm_inadequate_{'+'.join(which)}: {reason}"
        if (i + 1) % 10 == 0 or i == len(ambiguous_ids) - 1:
            print(f"  [B1-LLM] {i+1}/{len(ambiguous_ids)}", flush=True)

    clear_remove_ids = set(df.loc[b1_clear_remove_mask, "id"])
    df = df.loc[~(df["id"].isin(clear_remove_ids) | df["id"].isin(llm_remove_ids))].copy()
    print(f"[B1] LLM: giu {len(llm_keep_ids)}, loai {len(llm_remove_ids)}  =>  con lai {len(df)}")

    # ---------- B2: directions du dai (bo qua cac id da duoc LLM phan xu o B1) ----------
    b2_candidates = ~df["id"].isin(llm_keep_ids)
    b2_mask = b2_candidates & (df["directions"].astype(str).str.len() < MIN_DIRECTIONS_CHARS)
    mark_removed(df.loc[b2_mask, "id"], "b2_directions_too_short")
    df = df.loc[~b2_mask].copy()
    print(f"[B2] loai {b2_mask.sum()} recipe directions qua ngan (<{MIN_DIRECTIONS_CHARS} ky tu), con lai {len(df)}")

    # ---------- B3: exact duplicate theo title (chuan hoa) + sorted(set(mapped_names)) ----------
    df["_title_norm"] = df["title"].apply(normalize_title)
    df["_dup_key"] = df["_title_norm"] + "||" + df["_mapped_names"].apply(
        lambda names: ",".join(sorted(set(n.lower() for n in names)))
    )
    dup_mask = df.duplicated(subset="_dup_key", keep="first")
    dup_groups = df.groupby("_dup_key")["id"].apply(list)
    for key, ids in dup_groups.items():
        if len(ids) > 1:
            keep_id = ids[0]
            for rid in ids[1:]:
                log[rid]["reason"] = f"b3_exact_duplicate_of_{keep_id}"
    mark_removed(df.loc[dup_mask, "id"], "b3_exact_duplicate")
    df = df.loc[~dup_mask].copy()
    print(f"[B3] loai {dup_mask.sum()} ban ghi trung tuyet doi (theo mapped_ingredients), con lai {len(df)}")

    # ---------- B4: mo rong blacklist khong-phai-am-thuc ----------
    b4_mask = df["title"].str.contains(NONFOOD_RE)
    mark_removed(df.loc[b4_mask, "id"], "b4_nonfood_blacklist")
    df = df.loc[~b4_mask].copy()
    print(f"[B4] loai {b4_mask.sum()} recipe khong thuoc am thuc (blacklist mo rong), con lai {len(df)}")

    # ---------- B5: recipe qua chung chung ----------
    b5_mask = df["title"].str.contains(GENERIC_MEAL_RE) & (df["_n_unique"] > GENERIC_MAX_INGREDIENTS)
    mark_removed(df.loc[b5_mask, "id"], "b5_generic_collection")
    df = df.loc[~b5_mask].copy()
    print(f"[B5] loai {b5_mask.sum()} recipe qua chung chung (menu/collection + >12 nguyen lieu), con lai {len(df)}")

    # ---------- B6: near-duplicate fuzzy chat (title_sim>95 AND jaccard>0.9) ----------
    df["_mapped_name_set"] = df["_mapped_names"].apply(lambda names: frozenset(n.lower() for n in names))

    def block_key(title_norm):
        tokens = title_norm.split()
        return (tokens[0], tokens[-1]) if tokens else ("", "")

    df["_block_key"] = df["_title_norm"].apply(block_key)
    blocks = defaultdict(list)
    for rid, key in zip(df["id"], df["_block_key"]):
        blocks[key].append(rid)

    row_by_id = df.set_index("id")
    to_remove = set()

    for key, ids in tqdm(blocks.items(), desc="B6 fuzzy blocks"):
        if len(ids) < 2:
            continue
        for id_a, id_b in combinations(ids, 2):
            if id_a in to_remove or id_b in to_remove:
                continue
            title_a = row_by_id.at[id_a, "_title_norm"]
            title_b = row_by_id.at[id_b, "_title_norm"]
            title_sim = fuzz.token_sort_ratio(title_a, title_b)
            if title_sim <= FUZZY_TITLE_THRESHOLD:
                continue
            set_a = row_by_id.at[id_a, "_mapped_name_set"]
            set_b = row_by_id.at[id_b, "_mapped_name_set"]
            union = set_a | set_b
            jaccard = len(set_a & set_b) / len(union) if union else 0.0
            if jaccard > FUZZY_JACCARD_THRESHOLD:
                to_remove.add(id_b)
                log[id_b]["reason"] = (
                    f"b6_near_duplicate_of_{id_a}(title_sim={title_sim:.0f},jaccard={jaccard:.2f})"
                )

    for rid in to_remove:
        if log[rid]["status"] == "kept":
            log[rid]["status"] = "removed"
    df = df.loc[~df["id"].isin(to_remove)].copy()
    print(f"[B6] loai {len(to_remove)} ban ghi near-duplicate (title_sim>{FUZZY_TITLE_THRESHOLD}, "
          f"jaccard>{FUZZY_JACCARD_THRESHOLD}), con lai {len(df)}")

    # ---------- Xuat ket qua ----------
    output_cols = [c for c in pd.read_csv(INPUT_CSV, nrows=0).columns]
    df[output_cols].to_csv(OUTPUT_CSV, index=False)

    log_df = pd.DataFrame([
        {"id": rid, "title": info["title"], "status": info["status"], "reason": info["reason"]}
        for rid, info in log.items()
    ])
    log_df.to_csv(LOG_CSV, index=False)

    kept = (log_df["status"] == "kept").sum()
    removed = (log_df["status"] == "removed").sum()
    print()
    print(f"Tong dong ban dau (stage 2): {total_start}")
    print(f"Giu lai: {kept}  |  Loai bo: {removed}")
    print(f"Da ghi: {OUTPUT_CSV}, {LOG_CSV}")


if __name__ == "__main__":
    main()
