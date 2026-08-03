"""
Loc final_recipes.csv: chi giu lai recipe co >= MATCH_THRESHOLD % nguyen lieu (cot 'ner')
co the nhan dien duoc (match, khong can 100%) trong danh sach 'Natural Name' cua ingredients.csv.

Cach match moi ner-term:
  1. Match tuyet doi (term == mot Natural Name)
  2. Match theo tu (word-overlap): term co it nhat 1 tu (>=3 ky tu, khong tinh stopword) xuat
     hien trong tap tu vung duoc rut ra tu toan bo Natural Name (vd "chicken" trong "raw
     chicken breast" -> match qua tu "chicken")
  3. Fallback: fuzzy match (RapidFuzz) tu chinh cua term voi tu vung, cutoff cao, de bat loi
     chinh ta nhe (vd "seasame" -> "sesame")

Output:
  - recipes_filtered.csv : cac recipe dat nguong
  - recipes_filtered_log.csv : id, title, match_pct, matched/total ner terms, status
"""

import ast
import json
import re

import pandas as pd
from rapidfuzz import process, fuzz

INGREDIENTS_CSV = "ingredients.csv"
INPUT_CSV = "final_recipes.csv"
OUTPUT_CSV = "recipes_filtered.csv"
LOG_CSV = "recipes_filtered_log.csv"

MATCH_THRESHOLD = 0.70
FUZZY_CUTOFF = 85
STOPWORDS = {"of", "and", "with", "a", "the", "in", "or", "for", "to", "on", "no", "not"}


def words_of(s):
    return [w for w in re.findall(r"[a-z']+", str(s).lower()) if len(w) >= 3 and w not in STOPWORDS]


def parse_list_field(raw):
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    return items if isinstance(items, list) else []


def build_vocab(ingredients_csv):
    ing = pd.read_csv(ingredients_csv)
    natural_names = ing["Natural Name"].dropna().astype(str).str.strip().str.lower().unique().tolist()
    vocab = set()
    for nm in natural_names:
        vocab.update(words_of(nm))
    return vocab


def make_term_matcher(vocab):
    vocab_list = list(vocab)

    def is_matched(term):
        ws = words_of(term)
        if not ws:
            return False
        for w in ws:
            if w in vocab:
                return True
        longest = max(ws, key=len)
        match = process.extractOne(longest, vocab_list, scorer=fuzz.ratio, score_cutoff=FUZZY_CUTOFF)
        return match is not None

    return is_matched


def main():
    print("Dang xay dung tu vung tu ingredients.csv ...")
    vocab = build_vocab(INGREDIENTS_CSV)
    print(f"Tu vung: {len(vocab)} tu")
    is_matched = make_term_matcher(vocab)

    df = pd.read_csv(INPUT_CSV)
    print(f"Tong so recipe: {len(df)}")

    df["_ner"] = df["ner"].apply(parse_list_field)

    all_terms = set()
    for terms in df["_ner"]:
        for t in terms:
            all_terms.add(str(t).strip().lower())
    print(f"So ner-term duy nhat: {len(all_terms)}")

    print("Dang match tung term (1 lan / term, cache lai) ...")
    term_match_cache = {t: is_matched(t) for t in all_terms}
    n_matched_terms = sum(term_match_cache.values())
    print(f"So term match duoc: {n_matched_terms}/{len(all_terms)} "
          f"({n_matched_terms / len(all_terms) * 100:.1f}%)")

    def recipe_stats(terms):
        norm = [str(t).strip().lower() for t in terms]
        total = len(norm)
        if total == 0:
            return 0, 0, 0.0
        matched = sum(1 for t in norm if term_match_cache.get(t, False))
        return matched, total, matched / total

    stats = df["_ner"].apply(recipe_stats)
    df["_matched"] = stats.apply(lambda t: t[0])
    df["_total"] = stats.apply(lambda t: t[1])
    df["_match_pct"] = stats.apply(lambda t: t[2])

    keep_mask = df["_match_pct"] >= MATCH_THRESHOLD

    log_df = df[["id", "title", "_matched", "_total", "_match_pct"]].copy()
    log_df.columns = ["id", "title", "matched_ingredients", "total_ingredients", "match_pct"]
    log_df["status"] = keep_mask.map({True: "kept", False: "removed"})
    log_df.to_csv(LOG_CSV, index=False)

    output_cols = [c for c in pd.read_csv(INPUT_CSV, nrows=0).columns]
    df.loc[keep_mask, output_cols].to_csv(OUTPUT_CSV, index=False)

    print(f"\nNguong: >= {MATCH_THRESHOLD * 100:.0f}% ner-ingredient phai match")
    print(f"Giu lai: {keep_mask.sum()}  |  Loai bo: {(~keep_mask).sum()}")
    print(f"Da ghi: {OUTPUT_CSV}, {LOG_CSV}")


if __name__ == "__main__":
    main()
