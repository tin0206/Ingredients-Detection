from datasets import load_dataset
from collections import Counter

ds = load_dataset("Scuccorese/food-ingredients-dataset", split="train")

# Sử dụng Counter để đếm số lượng ảnh cho mỗi ingredient
ingredient_counts = Counter(ds["ingredient"])

# Sắp xếp theo tên ingredient để dễ theo dõi (giống cách bạn dùng sorted trước đó)
sorted_ingredients = sorted(ingredient_counts.items())

print(f"{'Ingredient':<30} | {'Count':<10}")
print("-" * 45)

for ingredient, count in sorted_ingredients:
    print(f"{ingredient:<30} | {count:<10}")