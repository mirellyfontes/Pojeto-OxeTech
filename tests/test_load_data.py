from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import load_data


def test_dataset_exists_and_has_expected_folders():
    assert Path(load_data.DATASET_DIR).exists(), f"Dataset directory not found: {load_data.DATASET_DIR}"
    assert any(Path(load_data.DATASET_DIR).iterdir()), "Dataset directory is empty"


def test_count_images_per_class_returns_non_empty_mapping():
    counts = load_data.contar_imagens_por_classe()
    assert counts, "No class counts were returned"
    assert all(value >= 0 for value in counts.values())
