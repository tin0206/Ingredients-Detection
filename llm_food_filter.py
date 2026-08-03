"""
Dung Ollama (local LLM, chay tren GPU) de quet toan bo final_recipes_filtered.csv
va tim them cac recipe KHONG PHAI mon an/do uong thuc su (my pham, do lam sach,
do choi trang tri, do thu cung that su, ...) ma bo loc tu-khoa da bo sot
(vd: Bath Bombs, Herbal Foot Soak, Diy Deodorant, Brown Sugar Body Scrub).

Thiet ke 2 pha de can bang toc do / do chinh xac:
  Pha 1 (nhanh, batch 8 recipe/lan): quet toan bo du lieu, tim ung vien nghi ngo
        (is_food=False). Da test: bat dung cac case xau that, nhung con ~5-7%
        false positive tren mon an that.
  Pha 2 (cham hon, tung recipe mot): verify lai CHI rieng cac ung vien bi pha 1
        gan is_food=False, de loc bot false positive truoc khi dua vao review.

Khong tu dong xoa recipe nao - chi ghi ket qua ra file de kiem tra/merge vao
manual_review.csv (buoc 9 cua pipeline loc chinh, filter_final_recipes_30k.py).

Co the dung giua chung va chay lai: script tu doc file ket qua da co de resume,
bo qua id da xu ly roi.
"""

import ast
import json
import sys
import time

import pandas as pd
import requests

INPUT_CSV = "final_recipes_filtered.csv"
PASS1_OUTPUT = "llm_pass1_results.csv"
PASS2_OUTPUT = "llm_pass2_verified.csv"
REVIEW_CSV = "manual_review.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"
PASS1_BATCH_SIZE = 8
N_INGREDIENTS_SHOWN = 6
MAX_RETRIES = 3

PASS1_PROMPT_HEADER = """For each recipe, is it meant to be EATEN or DRUNK by a human (is_food=true), or something else like skincare/cosmetics/cleaning/candles/non-edible-craft/literal pet food (is_food=false)?
Judge each recipe independently and carefully based on its own ingredients - do not let other recipes in this list influence your answer.
Double check: if your reason describes something edible/drinkable, is_food MUST be true.

Recipes:
"""

PASS1_PROMPT_FOOTER = """
Respond with ONLY a JSON array, no markdown, no extra text, one object per recipe above in the same order:
[{"id": <id>, "is_food": <bool>, "reason": "<short reason>"}]
"""

PASS2_PROMPT_TEMPLATE = """Is this recipe meant to be EATEN or DRUNK by a human (is_food=true), or is it something else like skincare/cosmetics/cleaning/candles/non-edible-craft/literal pet food (is_food=false)?
Note: joke titles like "Dog Food"/"Cat Food"/"Puppy Chow" that are actually eaten by humans should be is_food=true. Judge mainly by the ingredients.

title="{title}" ingredients={ingredients}
"""

PASS2_SCHEMA = {
    "type": "object",
    "properties": {
        "is_food": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_food", "reason"],
}

session = requests.Session()


def ollama_generate(prompt, schema=None, timeout=120):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    if schema is not None:
        payload["format"] = schema
    r = session.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


def parse_json_array(text):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array found in response")
    return json.loads(text[start:end + 1])


def first_ingredients(raw_ingredients):
    try:
        items = json.loads(raw_ingredients)
    except (TypeError, json.JSONDecodeError):
        try:
            items = ast.literal_eval(raw_ingredients)
        except (ValueError, SyntaxError):
            items = []
    return items[:N_INGREDIENTS_SHOWN] if isinstance(items, list) else []


def build_pass1_prompt(batch_df):
    lines = []
    for _, row in batch_df.iterrows():
        ing = first_ingredients(row["ingredients"])
        lines.append(f"id={row['id']} title=\"{row['title']}\" ingredients={json.dumps(ing)}")
    return PASS1_PROMPT_HEADER + "\n".join(lines) + PASS1_PROMPT_FOOTER


def _valid_pass1_results(results, expected_ids):
    if not isinstance(results, list):
        return False
    got_ids = set()
    for r in results:
        if not isinstance(r, dict) or "id" not in r or "is_food" not in r:
            return False
        if not isinstance(r["is_food"], bool):
            return False
        got_ids.add(r["id"])
    return got_ids == expected_ids


def run_pass1_batch(batch_df):
    prompt = build_pass1_prompt(batch_df)
    expected_ids = set(batch_df["id"].tolist())
    for attempt in range(MAX_RETRIES):
        try:
            text = ollama_generate(prompt)
            results = parse_json_array(text)
            if _valid_pass1_results(results, expected_ids):
                return results
            print(f"  [warn] batch result invalid/mismatched (attempt {attempt+1}), retrying", flush=True)
        except Exception as e:
            print(f"  [warn] pass1 batch failed (attempt {attempt+1}): {e}", flush=True)
        time.sleep(2)
    # fallback: classify one-by-one for this batch
    print("  [fallback] classifying batch items individually", flush=True)
    return [run_pass2_single(row) for _, row in batch_df.iterrows()]


def run_pass2_single(row):
    ing = first_ingredients(row["ingredients"])
    prompt = PASS2_PROMPT_TEMPLATE.format(title=row["title"], ingredients=json.dumps(ing))
    for attempt in range(MAX_RETRIES):
        try:
            text = ollama_generate(prompt, schema=PASS2_SCHEMA)
            data = json.loads(text)
            return {"id": row["id"], "is_food": bool(data["is_food"]), "reason": data.get("reason", "")}
        except Exception as e:
            print(f"  [warn] pass2 item {row['id']} failed (attempt {attempt+1}): {e}", flush=True)
            time.sleep(2)
    return {"id": row["id"], "is_food": True, "reason": "error_defaulted_true"}


def append_csv(path, rows, columns):
    df = pd.DataFrame(rows, columns=columns)
    header = not _file_has_content(path)
    df.to_csv(path, mode="a", header=header, index=False)


def _file_has_content(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return bool(f.readline())
    except FileNotFoundError:
        return False


def load_processed_ids(path):
    try:
        return set(pd.read_csv(path)["id"].tolist())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return set()


def run_pass1(df):
    done_ids = load_processed_ids(PASS1_OUTPUT)
    remaining = df[~df["id"].isin(done_ids)]
    total = len(df)
    print(f"[pass1] {len(done_ids)}/{total} da xu ly truoc do, con lai {len(remaining)}", flush=True)

    batches = [remaining.iloc[i:i + PASS1_BATCH_SIZE] for i in range(0, len(remaining), PASS1_BATCH_SIZE)]
    t0 = time.time()
    for i, batch in enumerate(batches):
        try:
            results = run_pass1_batch(batch)
            title_by_id = dict(zip(batch["id"], batch["title"]))
            rows = [{"id": r["id"], "title": title_by_id.get(r["id"], ""),
                      "is_food": r["is_food"], "reason": r.get("reason", "")} for r in results
                     if "id" in r and "is_food" in r]
            missing = set(batch["id"]) - {r["id"] for r in rows}
            for mid in missing:
                rows.append({"id": mid, "title": title_by_id.get(mid, ""),
                             "is_food": True, "reason": "error_defaulted_true"})
            append_csv(PASS1_OUTPUT, rows, ["id", "title", "is_food", "reason"])
        except Exception as e:
            print(f"  [error] batch {i+1} failed entirely, will retry on next resume: {e}", flush=True)
            continue

        done_so_far = len(done_ids) + (i + 1) * PASS1_BATCH_SIZE
        elapsed = time.time() - t0
        rate = elapsed / max(1, (i + 1) * PASS1_BATCH_SIZE)
        remaining_items = max(0, len(remaining) - (i + 1) * PASS1_BATCH_SIZE)
        eta_min = remaining_items * rate / 60
        if i % 20 == 0 or i == len(batches) - 1:
            print(f"[pass1] batch {i+1}/{len(batches)}  (~{min(done_so_far, total)}/{total} recipes)  "
                  f"{rate:.2f}s/item  ETA {eta_min:.0f} min", flush=True)

    print("[pass1] hoan tat", flush=True)


def run_pass2():
    pass1_df = pd.read_csv(PASS1_OUTPUT)
    flagged = pass1_df[pass1_df["is_food"] == False]  # noqa: E712
    done_ids = load_processed_ids(PASS2_OUTPUT)
    remaining = flagged[~flagged["id"].isin(done_ids)]
    print(f"[pass2] {len(flagged)} ung vien tu pass1, {len(done_ids)} da verify, "
          f"con lai {len(remaining)}", flush=True)

    if len(remaining) == 0:
        print("[pass2] khong con gi de verify", flush=True)
        return

    full_df = pd.read_csv(INPUT_CSV)
    full_df = full_df.set_index("id")

    t0 = time.time()
    for i, (_, row) in enumerate(remaining.iterrows()):
        try:
            full_row = full_df.loc[row["id"]]
            if isinstance(full_row, pd.DataFrame):
                full_row = full_row.iloc[0]
            result = run_pass2_single({"id": row["id"], "title": row["title"],
                                        "ingredients": full_row["ingredients"]})
            append_csv(PASS2_OUTPUT, [result], ["id", "is_food", "reason"])
        except Exception as e:
            print(f"  [error] pass2 item {row['id']} failed entirely, skipping: {e}", flush=True)
            continue
        if i % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta_min = (len(remaining) - i - 1) * rate / 60
            print(f"[pass2] {i+1}/{len(remaining)}  {rate:.2f}s/item  ETA {eta_min:.0f} min", flush=True)

    print("[pass2] hoan tat", flush=True)


def merge_into_review():
    pass1_df = pd.read_csv(PASS1_OUTPUT)
    pass2_df = pd.read_csv(PASS2_OUTPUT) if _file_has_content(PASS2_OUTPUT) else pd.DataFrame(
        columns=["id", "is_food", "reason"])

    title_by_id = dict(zip(pass1_df["id"], pass1_df["title"]))
    verified_false = pass2_df[pass2_df["is_food"] == False]  # noqa: E712

    review_rows = []
    for _, row in verified_false.iterrows():
        review_rows.append({
            "id": row["id"],
            "title": title_by_id.get(row["id"], ""),
            "type": "nonfood",
            "confidence": "llm_verified_needs_review",
            "detail": row["reason"],
        })

    review_df = pd.read_csv(REVIEW_CSV) if _file_has_content(REVIEW_CSV) else pd.DataFrame(
        columns=["id", "title", "type", "confidence", "detail"])
    existing_ids = set(review_df["id"]) if len(review_df) else set()
    new_rows = [r for r in review_rows if r["id"] not in existing_ids]

    if new_rows:
        combined = pd.concat([review_df, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(REVIEW_CSV, index=False)

    print(f"[merge] pass1 flagged={len(pass1_df[pass1_df['is_food'] == False])}  "  # noqa: E712
          f"pass2 verified_false={len(verified_false)}  them moi vao {REVIEW_CSV}: {len(new_rows)}", flush=True)


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Tong so recipe can quet: {len(df)}", flush=True)

    run_pass1(df)
    run_pass2()
    merge_into_review()

    print("Xong toan bo LLM food-check.", flush=True)


if __name__ == "__main__":
    main()
