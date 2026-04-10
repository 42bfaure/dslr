import csv
import utils
import sys
from typing import List, Any, Dict

from sklearn.metrics import accuracy_score


def _standardize_row(raw: List[float], standardization: List[List[float]]) -> List[float]:
	out: List[float] = []
	for j in range(len(raw)):
		mean, stddev = standardization[j][0], standardization[j][1]
		if stddev == 0.0 or abs(stddev) < 1e-15:
			out.append(0.0)
		else:
			out.append((raw[j] - mean) / stddev)
	return out


def _predict_row(x_std: List[float], thetas: List[List[float]], house_labels: List[str]) -> str:
	best_k = 0
	best_score = -1.0
	for k, theta in enumerate(thetas):
		z = theta[0]
		for j in range(len(x_std)):
			z += theta[j + 1] * x_std[j]
		s = utils.sigmoid(z)
		if s > best_score:
			best_score = s
			best_k = k
	return house_labels[best_k]


def predict_houses_in_row_order(model: Dict[str, Any], header: List[str], data: List[List]) -> List[str]:
	"""Une prédiction par ligne du CSV (ordre des lignes), pour houses.csv et métriques."""
	features: List[str] = model["features"]
	standardization: List[List[float]] = model["standardization"]
	thetas: List[List[float]] = model["thetas"]
	house_labels: List[str] = model["house_labels"]

	features_indices = {name: header.index(name) for name in features}
	index_col = header.index("Index")
	n_rows = len(data[index_col])

	out: List[str] = []
	for i in range(n_rows):
		raw: List[float] = []
		for j, fname in enumerate(features):
			idx = features_indices[fname]
			val = data[idx][i]
			if val is None:
				raw.append(float(standardization[j][0]))
			else:
				raw.append(float(val))
		x_std = _standardize_row(raw, standardization)
		out.append(_predict_row(x_std, thetas, house_labels))
	return out


def predict_from_model(model: Dict[str, Any], header: List[str], data: List[List]) -> List[tuple]:
	index_col = header.index("Index")
	houses_per_row = predict_houses_in_row_order(model, header, data)
	n_rows = len(data[index_col])
	return [(data[index_col][i], houses_per_row[i]) for i in range(n_rows)]


def print_accuracy_if_labels(header: List[str], data: List[List], predicted_houses: List[str]) -> None:
	"""Affiche l'accuracy sklearn dans le terminal si la colonne Hogwarts House contient des étiquettes."""
	if "Hogwarts House" not in header:
		print("Accuracy: colonne « Hogwarts House » absente — non calculée.", flush=True)
		return
	hi = header.index("Hogwarts House")
	n = len(predicted_houses)
	y_true: List[str] = []
	y_pred: List[str] = []
	for i in range(n):
		true_h = data[hi][i]
		if true_h is None or (isinstance(true_h, str) and true_h.strip() == ""):
			continue
		y_true.append(str(true_h))
		y_pred.append(predicted_houses[i])
	if len(y_true) == 0:
		print("Accuracy: aucune étiquette connue dans le CSV — non calculée.", flush=True)
		return
	acc = accuracy_score(y_true, y_pred)
	print(f"Accuracy (scikit-learn): {acc:.4f}  (sur {len(y_true)} exemple(s) étiquetés)", flush=True)


def write_houses_csv(path: str, predictions: List[tuple]) -> None:
	sorted_rows = sorted(predictions, key=lambda r: int(r[0]) if r[0] is not None else 0)
	with open(path, "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["Index", "Hogwarts House"])
		for index_val, house in sorted_rows:
			w.writerow([index_val, house])


def main() -> bool:
	print("Starting main")
	try:
		if len(sys.argv) != 3:
			raise AttributeError("You must input 2 paths")
		data_path = sys.argv[1]
		model_path = sys.argv[2]
		model = utils.load_logreg_model_json(model_path)
		if model is None:
			raise RuntimeError("Impossible de charger le modèle")
		loaded = utils.load_csv(data_path, return_header=True, auto_convert=True)
		if loaded is None:
			raise RuntimeError("Impossible de charger les données")
		header, data = loaded
		houses_per_row = predict_houses_in_row_order(model, header, data)
		print_accuracy_if_labels(header, data, houses_per_row)
		index_col = header.index("Index")
		predictions = [(data[index_col][i], houses_per_row[i]) for i in range(len(houses_per_row))]
		out_path = "houses.csv"
		write_houses_csv(out_path, predictions)
		print(f"Prédictions écrites dans {out_path}", flush=True)
	except Exception as e:
		print(f"Error: {e}")
		return False
	print("Ending main")
	return True


if __name__ == "__main__":
	main()
