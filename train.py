from pathlib import Path
import sys

def ensure_dir(p: Path):
	if not p.exists():
		raise FileNotFoundError(f"Missing directory: {p}")

def build_dataset_yaml(root: Path, names_count: int = 120) -> Path:
	train_images = root / "train" / "images"
	val_images = root / "valid" / "images"
	test_images = root / "test" / "images"
	ensure_dir(train_images)
	ensure_dir(val_images)
	ensure_dir(test_images)
	names = [f"ingredient_{i}" for i in range(names_count)]
	yaml_text = (
		"path: .\n"
		f"train: {train_images.as_posix()}\n"
		f"val: {val_images.as_posix()}\n"
		f"test: {test_images.as_posix()}\n"
		f"names: [{', '.join(names)}]\n"
	)
	yaml_path = root / "dataset.yaml"
	yaml_path.write_text(yaml_text, encoding="utf-8")
	return yaml_path

def select_device():
    import torch
    if torch.cuda.is_available():
        return 0
    return "cpu"

def train_yolo(root: Path):
	try:
		from ultralytics import YOLO
	except ImportError:
		print("ultralytics not found. Install with: python -m pip install ultralytics")
		sys.exit(1)
	data_yaml = build_dataset_yaml(root, names_count=120)
	device = select_device()
	model = YOLO("yolov8n.pt")
	model.train(
		data=str(data_yaml),
		epochs=50,
		imgsz=640,
		batch=16,
		device=0,
		workers=4,
		project=str(root / "runs"),
		name="ingredients-yolov8n",
	)

if __name__ == "__main__":
	project_root = Path(__file__).resolve().parent
	train_yolo(project_root)
