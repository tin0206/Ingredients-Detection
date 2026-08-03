"""
Loc final_recipes.csv bang LLM (Ollama, qwen2.5:7b-instruct chay tren GPU) theo 3 tieu chi:

  1. Loai cong thuc khong thuoc ve recipe that (nuoc ngam chan, son mong tay, my pham, ...)
  2. Loai cong thuc qua chung chung (mot bua an gom nhieu mon, khong phai 1 mon cu the)
  3. Loai ban trung: phat hien cap/nhom gan giong nhau (RapidFuzz tren title + jaccard tren
     mapped_ingredients), roi dung LLM so sanh ca 2 ban (ingredients, directions, servings...)
     de giu lai ban DAY DU/RO RANG hon, khong chi giu ban dau tien.

Thiet ke resumable: moi ket qua duoc ghi ngay ra CSV sau moi item/batch, chay lai se tu
bo qua phan da xong.

Output:
  - final_recipes_v1.csv   : ket qua sau khi loc
  - filter_log_v1.csv      : id, title, status (kept/removed), reason
"""

import ast
import json
import re
import time
from collections import defaultdict
from itertools import combinations

import pandas as pd
import requests
from rapidfuzz import fuzz
from tqdm import tqdm

INPUT_CSV = "final_recipes.csv"
OUTPUT_CSV = "final_recipes_v1.csv"
LOG_CSV = "filter_log_v1.csv"

DEDUP_RESULTS_CSV = "dedup_pairs_results.csv"
CLASSIFY_PASS1_CSV = "classify_pass1_results.csv"
CLASSIFY_PASS2_CSV = "classify_pass2_verified.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

# ---- dedup candidate-pair thresholds (net rong, LLM se la trong tai cuoi cung) ----
# Neu title trung gan tuyet doi (>= CANDIDATE_TITLE_SIM_STRICT) thi luon coi la ung vien,
# khong doi hoi jaccard nua - vi mapped_ingredients co the anh xa ten USDA rat khac nhau
# ngay ca khi 2 recipe la cung 1 mon (da xac nhan qua test thuc te voi "Chicken Tortilla
# Soup" vs "Chicken Tortilla Soup Recipe": title_sim=100 nhung jaccard chi 0.18).
CANDIDATE_TITLE_SIM_STRICT = 95
CANDIDATE_TITLE_SIM = 80
CANDIDATE_JACCARD = 0.3

TITLE_PREFIXES = [
    "easy", "best", "the best", "ultimate", "quick", "quick and easy",
    "classic", "simple", "perfect", "homemade", "old fashioned",
    "old-fashioned", "traditional", "authentic", "famous", "super",
    "delicious", "amazing", "yummy", "favorite", "favourite",
    "award winning", "world's best", "worlds best", "no-fail", "nofail",
    "foolproof", "grandma's", "grandmas", "mom's", "moms", "my",
]
TITLE_SUFFIXES = ["recipe", "recipes"]


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


def parse_list_field(raw):
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return items if isinstance(items, list) else []


def mapped_names(raw):
    items = parse_list_field(raw)
    return [str(e["mapped_name"]).strip() for e in items if isinstance(e, dict) and e.get("mapped_name")]


def ollama_generate(prompt, schema=None, timeout=90):
    payload = {"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    if schema is not None:
        payload["format"] = schema
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def file_has_content(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return bool(f.readline())
    except FileNotFoundError:
        return False


def append_csv(path, rows, columns):
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, mode="a", header=not file_has_content(path), index=False)


def load_done_keys(path, key_col):
    if not file_has_content(path):
        return set()
    return set(pd.read_csv(path)[key_col].astype(str).tolist())


# ============================== Buoc 1: tim cap trung nghi ngo ==============================

def find_candidate_pairs(df):
    df = df.copy()
    df["_title_norm"] = df["title"].apply(normalize_title)
    df["_mapped_set"] = df["mapped_ingredients"].apply(lambda r: frozenset(n.lower() for n in mapped_names(r)))

    def block_key(t):
        tok = t.split()
        return (tok[0], tok[-1]) if tok else ("", "")

    df["_block"] = df["_title_norm"].apply(block_key)
    blocks = defaultdict(list)
    for rid, key in zip(df["id"], df["_block"]):
        blocks[key].append(rid)

    row = df.set_index("id")
    pairs = []
    for key, ids in blocks.items():
        if len(ids) < 2:
            continue
        for a, b in combinations(ids, 2):
            ta, tb = row.at[a, "_title_norm"], row.at[b, "_title_norm"]
            sim = fuzz.token_sort_ratio(ta, tb)
            if sim < CANDIDATE_TITLE_SIM:
                continue
            sa, sb = row.at[a, "_mapped_set"], row.at[b, "_mapped_set"]
            union = sa | sb
            jac = len(sa & sb) / len(union) if union else 0.0
            if sim < CANDIDATE_TITLE_SIM_STRICT and jac < CANDIDATE_JACCARD:
                continue
            pairs.append((int(a), int(b), sim, jac))
    return pairs


DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_same_dish": {"type": "boolean"},
        "keep": {"type": "string", "enum": ["A", "B"]},
        "reason": {"type": "string"},
    },
    "required": ["is_same_dish", "keep", "reason"],
}

DEDUP_PROMPT_TEMPLATE = """Compare these two recipes. Are they duplicates / variations of the same dish
(possibly with a different title wording, e.g. "Hamburger Stew" vs "Beef Hamburger Stew")?

is_same_dish: true only if they are genuinely the same dish (not just two different recipes that
happen to share a common word or category).
keep: if is_same_dish is true, which one (A or B) has MORE COMPLETE / CLEARER information overall
(more specific ingredient amounts, clearer step-by-step directions, serving info present)? If both
are equally good, prefer A.

Recipe A:
title="{title_a}"
ingredients={ingredients_a}
directions={directions_a}
estimated_servings={servings_a}

Recipe B:
title="{title_b}"
ingredients={ingredients_b}
directions={directions_b}
estimated_servings={servings_b}
"""


def resolve_duplicate(row_a, row_b, max_retries=3):
    prompt = DEDUP_PROMPT_TEMPLATE.format(
        title_a=row_a["title"], ingredients_a=json.dumps(parse_list_field(row_a["ingredients"])[:15]),
        directions_a=json.dumps(parse_list_field(row_a["directions"])[:8]), servings_a=row_a["estimated_servings"],
        title_b=row_b["title"], ingredients_b=json.dumps(parse_list_field(row_b["ingredients"])[:15]),
        directions_b=json.dumps(parse_list_field(row_b["directions"])[:8]), servings_b=row_b["estimated_servings"],
    )
    for attempt in range(max_retries):
        try:
            text = ollama_generate(prompt, schema=DEDUP_SCHEMA)
            data = json.loads(text)
            return bool(data["is_same_dish"]), data["keep"], data.get("reason", "")
        except Exception as e:
            print(f"  [warn] dedup compare failed (attempt {attempt+1}): {e}", flush=True)
            time.sleep(2)
    return False, "A", "error_defaulted_not_duplicate"


def run_dedup_stage(df):
    pairs = find_candidate_pairs(df)
    print(f"[dedup] {len(pairs)} cap nghi ngo trung lap can LLM xac nhan", flush=True)

    done_keys = load_done_keys(DEDUP_RESULTS_CSV, "pair_key")
    row_by_id = df.set_index("id")

    to_process = [(a, b, sim, jac) for a, b, sim, jac in pairs if f"{a}-{b}" not in done_keys]
    print(f"[dedup] {len(pairs) - len(to_process)} da xu ly truoc do, con lai {len(to_process)}", flush=True)

    t0 = time.time()
    for i, (a, b, sim, jac) in enumerate(to_process):
        row_a, row_b = row_by_id.loc[a], row_by_id.loc[b]
        is_dup, keep, reason = resolve_duplicate(row_a, row_b)
        remove_id = b if keep == "A" else a
        append_csv(DEDUP_RESULTS_CSV, [{
            "pair_key": f"{a}-{b}", "id_a": a, "id_b": b, "title_a": row_a["title"], "title_b": row_b["title"],
            "title_sim": sim, "ingredient_jaccard": round(jac, 3),
            "is_same_dish": is_dup, "keep": keep, "remove_id": remove_id if is_dup else "",
            "reason": reason,
        }], ["pair_key", "id_a", "id_b", "title_a", "title_b", "title_sim", "ingredient_jaccard",
             "is_same_dish", "keep", "remove_id", "reason"])
        if (i + 1) % 20 == 0 or i == len(to_process) - 1:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta_min = (len(to_process) - i - 1) * rate / 60
            print(f"[dedup] {i+1}/{len(to_process)}  {rate:.1f}s/pair  ETA {eta_min:.0f} min", flush=True)

    print("[dedup] hoan tat", flush=True)


# ============================== Buoc 2: phan loai (khong-phai-recipe / qua chung chung) ==============================

CLASSIFY_PROMPT_HEADER = """For each recipe, classify it into exactly one category:
- "ok": a real, specific food/drink recipe for humans.
- "not_a_recipe": NOT an edible food/drink recipe (e.g. skincare/cosmetics like foot soak, nail
  polish, lip balm, bath bomb; cleaning products; non-edible crafts; literal pet food) - judge by
  ingredients, not just a playful title (e.g. joke titles like "Dog Food" that are actually eaten
  by humans should be "ok").
- "too_generic": describes a whole MEAL/MENU made of many different dishes rather than one
  specific dish (e.g. "Summer Lunch Spread", "Weekly Dinner Plan"), not just a dish with many
  ingredients.
Judge each recipe independently based on its own title/ingredients - do not let other recipes in
this list influence your answer.

Recipes:
"""

CLASSIFY_PROMPT_FOOTER = """
Respond with ONLY a JSON array, no markdown, no extra text, one object per recipe above in the
same order:
[{"id": <id>, "category": "ok"|"not_a_recipe"|"too_generic", "reason": "<short reason>"}]
"""

CLASSIFY_BATCH_SIZE = 8
VALID_CATEGORIES = {"ok", "not_a_recipe", "too_generic"}


def build_classify_prompt(batch_df):
    lines = []
    for _, row in batch_df.iterrows():
        ing = parse_list_field(row["ingredients"])[:6]
        lines.append(f"id={row['id']} title=\"{row['title']}\" ingredients={json.dumps(ing)}")
    return CLASSIFY_PROMPT_HEADER + "\n".join(lines) + CLASSIFY_PROMPT_FOOTER


def parse_json_array(text):
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in response")
    return json.loads(text[start:end + 1])


def valid_classify_results(results, expected_ids):
    if not isinstance(results, list):
        return False
    got = set()
    for r in results:
        if not isinstance(r, dict) or "id" not in r or r.get("category") not in VALID_CATEGORIES:
            return False
        got.add(r["id"])
    return got == expected_ids


def classify_batch(batch_df, max_retries=3):
    prompt = build_classify_prompt(batch_df)
    expected_ids = set(batch_df["id"].tolist())
    for attempt in range(max_retries):
        try:
            text = ollama_generate(prompt)
            results = parse_json_array(text)
            if valid_classify_results(results, expected_ids):
                return results
            print(f"  [warn] classify batch invalid (attempt {attempt+1}), retrying", flush=True)
        except Exception as e:
            print(f"  [warn] classify batch failed (attempt {attempt+1}): {e}", flush=True)
        time.sleep(2)
    print("  [fallback] classifying batch items individually", flush=True)
    out = []
    for _, row in batch_df.iterrows():
        out.extend(classify_single(row))
    return out


SINGLE_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["ok", "not_a_recipe", "too_generic"]},
        "reason": {"type": "string"},
    },
    "required": ["category", "reason"],
}


def classify_single(row, max_retries=3):
    ing = parse_list_field(row["ingredients"])[:6]
    prompt = (f'Classify this recipe: title="{row["title"]}" ingredients={json.dumps(ing)}\n\n'
              f'Categories: "ok" (real specific food/drink recipe), "not_a_recipe" (skincare/cosmetics/'
              f'cleaning/non-edible craft/literal pet food - judge by ingredients not just title), '
              f'"too_generic" (a whole meal/menu of many dishes, not one specific dish).')
    for attempt in range(max_retries):
        try:
            text = ollama_generate(prompt, schema=SINGLE_CLASSIFY_SCHEMA)
            data = json.loads(text)
            return [{"id": row["id"], "category": data["category"], "reason": data.get("reason", "")}]
        except Exception as e:
            print(f"    [warn] single classify failed (attempt {attempt+1}): {e}", flush=True)
            time.sleep(2)
    return [{"id": row["id"], "category": "ok", "reason": "error_defaulted_ok"}]


def run_classify_pass1(df):
    """Pha 1: quet nhanh theo batch de tim UNG VIEN nghi ngo (khong dang tin cay 100%)."""
    done_ids = load_done_keys(CLASSIFY_PASS1_CSV, "id")
    remaining = df[~df["id"].astype(str).isin(done_ids)]
    print(f"[classify-p1] {len(done_ids)} da xu ly truoc do, con lai {len(remaining)}", flush=True)

    batches = [remaining.iloc[i:i + CLASSIFY_BATCH_SIZE] for i in range(0, len(remaining), CLASSIFY_BATCH_SIZE)]
    t0 = time.time()
    for i, batch in enumerate(batches):
        title_by_id = dict(zip(batch["id"], batch["title"]))
        try:
            results = classify_batch(batch)
            rows = [{"id": r["id"], "title": title_by_id.get(r["id"], ""),
                      "category": r["category"], "reason": r.get("reason", "")} for r in results
                     if "id" in r and r.get("category") in VALID_CATEGORIES]
            missing = set(batch["id"]) - {r["id"] for r in rows}
            for mid in missing:
                rows.append({"id": mid, "title": title_by_id.get(mid, ""), "category": "ok",
                             "reason": "error_defaulted_ok"})
            append_csv(CLASSIFY_PASS1_CSV, rows, ["id", "title", "category", "reason"])
        except Exception as e:
            print(f"  [error] batch {i+1} failed entirely, will retry on next resume: {e}", flush=True)
            continue

        if (i + 1) % 20 == 0 or i == len(batches) - 1:
            elapsed = time.time() - t0
            rate = elapsed / max(1, (i + 1) * CLASSIFY_BATCH_SIZE)
            remaining_items = max(0, len(remaining) - (i + 1) * CLASSIFY_BATCH_SIZE)
            eta_min = remaining_items * rate / 60
            print(f"[classify-p1] batch {i+1}/{len(batches)}  {rate:.2f}s/item  ETA {eta_min:.0f} min", flush=True)

    print("[classify-p1] hoan tat", flush=True)


def run_classify_pass2(df):
    """Pha 2: verify TUNG CAI MOT (dang tin cay hon) chi cho cac ung vien pha 1 gan
    category != ok, de loc bot false positive truoc khi xoa that."""
    pass1 = pd.read_csv(CLASSIFY_PASS1_CSV)
    flagged = pass1[pass1["category"] != "ok"]
    done_ids = load_done_keys(CLASSIFY_PASS2_CSV, "id")
    remaining = flagged[~flagged["id"].astype(str).isin(done_ids)]
    print(f"[classify-p2] {len(flagged)} ung vien tu pha 1, {len(done_ids)} da verify, "
          f"con lai {len(remaining)}", flush=True)

    if len(remaining) == 0:
        print("[classify-p2] khong con gi de verify", flush=True)
        return

    row_by_id = df.set_index("id")
    t0 = time.time()
    for i, (_, r) in enumerate(remaining.iterrows()):
        rid = int(r["id"])
        try:
            row = row_by_id.loc[rid]
        except KeyError:
            continue
        row = row.copy()
        row["id"] = rid
        result = classify_single(row)[0]
        append_csv(CLASSIFY_PASS2_CSV, [result], ["id", "category", "reason"])
        if (i + 1) % 20 == 0 or i == len(remaining) - 1:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta_min = (len(remaining) - i - 1) * rate / 60
            print(f"[classify-p2] {i+1}/{len(remaining)}  {rate:.1f}s/item  ETA {eta_min:.0f} min", flush=True)

    print("[classify-p2] hoan tat", flush=True)


# ============================== Main ==============================

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Tong so recipe: {len(df)}", flush=True)

    run_dedup_stage(df)
    run_classify_pass1(df)
    run_classify_pass2(df)

    dedup_results = pd.read_csv(DEDUP_RESULTS_CSV) if file_has_content(DEDUP_RESULTS_CSV) else pd.DataFrame(
        columns=["pair_key", "id_a", "id_b", "title_a", "title_b", "title_sim", "ingredient_jaccard",
                 "is_same_dish", "keep", "remove_id", "reason"])
    pass1 = pd.read_csv(CLASSIFY_PASS1_CSV) if file_has_content(CLASSIFY_PASS1_CSV) else pd.DataFrame(
        columns=["id", "category", "reason"])
    pass2 = pd.read_csv(CLASSIFY_PASS2_CSV) if file_has_content(CLASSIFY_PASS2_CSV) else pd.DataFrame(
        columns=["id", "category", "reason"])

    # pass2 (da verify tung cai) ghi de ket qua pass1 (batch, kem tin cay hon) khi co
    final_category = dict(zip(pass1["id"].astype(int), pass1["category"]))
    final_reason = dict(zip(pass1["id"].astype(int), pass1["reason"]))
    for rid, cat, reason in zip(pass2["id"].astype(int), pass2["category"], pass2["reason"]):
        final_category[rid] = cat
        final_reason[rid] = reason

    log = {rid: {"title": title, "status": "kept", "reason": ""} for rid, title in zip(df["id"], df["title"])}

    dup_confirmed = dedup_results[dedup_results["is_same_dish"] == True]  # noqa: E712
    for _, r in dup_confirmed.iterrows():
        rid = int(r["remove_id"])
        winner = r["id_b"] if r["remove_id"] == r["id_a"] else r["id_a"]
        if log[rid]["status"] == "kept":
            log[rid]["status"] = "removed"
            log[rid]["reason"] = f"duplicate_of_{winner}: {r['reason']}"

    for rid, cat in final_category.items():
        if cat != "ok" and rid in log and log[rid]["status"] == "kept":
            log[rid]["status"] = "removed"
            log[rid]["reason"] = f"{cat}: {final_reason.get(rid, '')}"

    log_df = pd.DataFrame([
        {"id": rid, "title": info["title"], "status": info["status"], "reason": info["reason"]}
        for rid, info in log.items()
    ])
    log_df.to_csv(LOG_CSV, index=False)

    kept_ids = set(log_df.loc[log_df["status"] == "kept", "id"])
    df[df["id"].isin(kept_ids)].to_csv(OUTPUT_CSV, index=False)

    kept, removed = (log_df["status"] == "kept").sum(), (log_df["status"] == "removed").sum()
    print(f"\nTong: {len(df)}  Giu lai: {kept}  Loai bo: {removed}", flush=True)
    print(f"Da ghi: {OUTPUT_CSV}, {LOG_CSV}", flush=True)


if __name__ == "__main__":
    main()
