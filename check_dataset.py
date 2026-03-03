from datasets import load_dataset

ds = load_dataset("Scuccorese/food-ingredients-dataset", split="train")

# Get unique ingredients
ingredient_list = sorted(set(ds["ingredient"]))

for ingredient in ingredient_list:
    print(ingredient)