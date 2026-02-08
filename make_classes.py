from datasets import load_dataset

print("Loading dataset...")
ds = load_dataset("Scuccorese/food-ingredients-dataset", split="train")

ingredients = sorted(set(ds["ingredient"]))

print(f"Total classes: {len(ingredients)}")

with open("classes.txt", "w", encoding="utf-8") as f:
    for ing in ingredients:
        f.write(ing + "\n")

print("✅ classes.txt created")
