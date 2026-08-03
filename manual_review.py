"""
Cong cu review thu cong cac dong trong manual_review.csv (duoc tao boi
filter_final_recipes_30k.py va llm_food_filter.py).

Chay:
    python manual_review.py                    # review tat ca, uu tien nonfood/category truoc
    python manual_review.py --type nonfood      # chi review 1 loai: nonfood / category_title / near_duplicate_pair
    python manual_review.py --apply             # ghi cac id da quyet dinh "remove" vao confirmed_removals.txt

Voi moi dong, go:
    k = giu lai (khong xoa)
    r = xoa (dua vao confirmed_removals.txt khi chay --apply)
    s = bo qua, xem lai sau
    q = luu va thoat

Quyet dinh duoc luu ngay vao review_decisions.csv sau moi lan tra loi, nen co
the dung giua chung va chay lai ma khong mat tien do.
"""

import argparse
import ast
import json
import re
import sys

import pandas as pd

REVIEW_CSV = "manual_review.csv"
DECISIONS_CSV = "review_decisions.csv"
RECIPES_CSV = "final_recipes_filtered.csv"
CONFIRMED_REMOVALS_FILE = "confirmed_removals.txt"

TYPE_ORDER = ["nonfood", "category_title", "near_duplicate_pair"]
PAIR_DETAIL_RE = re.compile(
    r"paired_with_id=(\d+)\s+title='(.*?)'\s+title_sim=([\d.]+)\s+jaccard=([\d.]+)"
)


def load_recipes_index():
    df = pd.read_csv(RECIPES_CSV)
    df["id"] = df["id"].astype(int)
    return df.set_index("id")


def short_ingredients(raw, n=8):
    try:
        items = json.loads(raw)
    except (TypeError, json.JSONDecodeError, ValueError):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            items = []
    if not isinstance(items, list):
        return ""
    return ", ".join(str(x) for x in items[:n]) + (" ..." if len(items) > n else "")


def print_recipe_block(label, rid, recipes_idx):
    print(f"  {label} [id={rid}]", end="")
    if rid in recipes_idx.index:
        row = recipes_idx.loc[rid]
        print(f" \"{row['title']}\"")
        print(f"      ingredients: {short_ingredients(row['ingredients'])}")
    else:
        print("  (khong tim thay trong final_recipes_filtered.csv - co the da bi xoa)")


def load_decisions():
    try:
        df = pd.read_csv(DECISIONS_CSV, dtype={"id": int})
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["id", "type", "decision"])


def save_decision(rid, rtype, decision):
    row = pd.DataFrame([{"id": rid, "type": rtype, "decision": decision}])
    header = False
    try:
        with open(DECISIONS_CSV, "r", encoding="utf-8") as f:
            header = not bool(f.readline())
    except FileNotFoundError:
        header = True
    row.to_csv(DECISIONS_CSV, mode="a", header=header, index=False)


def review(type_filter):
    review_df = pd.read_csv(REVIEW_CSV)
    review_df["id"] = review_df["id"].astype(int)
    decisions = load_decisions()
    decided_ids = set(decisions["id"]) if len(decisions) else set()

    recipes_idx = load_recipes_index()
    n_already_removed = (~review_df["id"].isin(recipes_idx.index)).sum()
    review_df = review_df[review_df["id"].isin(recipes_idx.index)]
    if n_already_removed:
        print(f"({n_already_removed} dong bi bo qua vi da bi loai tu dong tu truoc, khong con trong "
              f"{RECIPES_CSV})")

    if type_filter:
        review_df = review_df[review_df["type"] == type_filter]
    else:
        review_df["_order"] = review_df["type"].apply(
            lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER))
        review_df = review_df.sort_values("_order")

    pending = review_df[~review_df["id"].isin(decided_ids)]
    total = len(review_df)
    print(f"Tong so dong: {total}  |  Da quyet dinh truoc do: {len(decided_ids & set(review_df['id']))}"
          f"  |  Con lai: {len(pending)}\n")

    if len(pending) == 0:
        print("Khong con dong nao can review (voi bo loc hien tai).")
        return

    n_done_this_session = 0

    for _, row in pending.iterrows():
        rid, title, rtype = int(row["id"]), row["title"], row["type"]
        detail = row.get("detail", "")
        if pd.isna(detail):
            detail = "(khong co ly do cu the)"
        print("=" * 78)
        print(f"[{rtype}]  id={rid}  \"{title}\"")
        if rtype == "nonfood":
            print(f"  Ly do bi nghi ngo: {detail}")
            print_recipe_block("Recipe", rid, recipes_idx)
        elif rtype == "category_title":
            print(f"  Ly do bi nghi ngo: {detail}")
            print_recipe_block("Recipe", rid, recipes_idx)
        elif rtype == "near_duplicate_pair":
            m = PAIR_DETAIL_RE.search(str(detail))
            print_recipe_block("Recipe A", rid, recipes_idx)
            if m:
                pid, ptitle, sim, jac = m.groups()
                print(f"  --- nghi ngo trung voi (title_sim={sim}, ingredient_jaccard={jac}) ---")
                print_recipe_block("Recipe B", int(pid), recipes_idx)
            else:
                print(f"  detail: {detail}")
        else:
            print(f"  detail: {detail}")

        while True:
            ans = input("\n  [k]eep / [r]emove nay / [s]kip / [q]uit: ").strip().lower()
            if ans in ("k", "r", "s", "q"):
                break
            print("  Vui long go k, r, s hoac q.")

        if ans == "q":
            print(f"\nDa luu tien do. Xem lai {n_done_this_session} dong trong phien nay.")
            return
        if ans == "s":
            continue

        decision = "remove" if ans == "r" else "keep"
        save_decision(rid, rtype, decision)
        n_done_this_session += 1

    print(f"\nDa review het! ({n_done_this_session} dong trong phien nay)")


def apply_decisions():
    decisions = load_decisions()
    to_remove = decisions[decisions["decision"] == "remove"]["id"].tolist()
    if not to_remove:
        print("Khong co id nao duoc danh dau 'remove'.")
        return

    try:
        with open(CONFIRMED_REMOVALS_FILE, "r", encoding="utf-8") as f:
            existing = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        existing = set()

    new_ids = {str(i) for i in to_remove}
    combined = existing | new_ids

    with open(CONFIRMED_REMOVALS_FILE, "w", encoding="utf-8") as f:
        for i in sorted(combined, key=lambda x: int(x)):
            f.write(i + "\n")

    print(f"Da ghi {len(new_ids)} id (them {len(new_ids - existing)} id moi) vao {CONFIRMED_REMOVALS_FILE}.")
    print("Chay lai 'python filter_final_recipes_30k.py' de ap dung xoa vao final_recipes_filtered.csv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=TYPE_ORDER, default=None,
                         help="chi review 1 loai (mac dinh: tat ca, uu tien nonfood/category truoc)")
    parser.add_argument("--apply", action="store_true",
                         help="ghi cac id 'remove' da quyet dinh vao confirmed_removals.txt")
    args = parser.parse_args()

    if args.apply:
        apply_decisions()
        return

    try:
        review(args.type)
    except (KeyboardInterrupt, EOFError):
        print("\n\nDa luu tien do, thoat.")
        sys.exit(0)


if __name__ == "__main__":
    main()
