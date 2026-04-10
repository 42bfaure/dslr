import utils
import sys
import os
from typing import List, Tuple

from sklearn.metrics import accuracy_score

def extract_data(path: str) -> Tuple[List[str], List[List], List[str]]:
	"""
	1) Charge le CSV (header + data au format par colonnes)
	2) Sélectionne des features peu corrélées

	Returns:
		(header, data, courses_names)
	"""
	print("Starting extract_data")
	header, data = utils.load_csv(path, columns=None, skip_header=True, return_header=True, auto_convert=True)
	if data is None:
		raise Exception("Failed to load data")

	numeric_names, numeric_data = utils.extract_numeric_columns(path, data, skip_header=True)
	if numeric_names is None or numeric_data is None:
		raise Exception("No numeric data found")

	selected_features = utils.select_least_correlated_features(
		numeric_names, numeric_data,
		max_correlation=0.7,
		exclude_columns=["Index", "Hogwarts House"]
	)
	if selected_features is None or len(selected_features) == 0:
		raise Exception("Aucune feature sélectionnée")

	courses_names = [name for name in selected_features if name != "Index"]
	print("Ending extract_data")
	return header, data, courses_names

def prepare_data(header: List[str], data: List[List], features_names: List[str]) -> Tuple[List[List[float]], List[int], int, List[str]]:
	"""
	Prépare les données au format ML standard:
	- X: List[List[float]] (m lignes, n features)
	- y: List[int] (m labels 0..3)
	- m: int (nb d'exemples)
	
	Args:
		header: En-têtes du CSV
		data: Données brutes (List[List])
		features_names: Liste des noms des features sélectionnées
	
	Returns:
		Tuple (X, y, m, house_labels) où:
		- X: Liste de listes, chaque ligne = features d'un étudiant
		- y: Liste de labels (0, 1, 2, 3 pour les 4 maisons)
		- m: Nombre total d'étudiants
		- house_labels: noms des maisons triés (indice = label)
	"""
	print("Starting prepare_data")
	houses_index = header.index("Hogwarts House")
	houses = sorted({h for h in data[houses_index] if h is not None})
	house_to_label = {house: idx for idx, house in enumerate(houses)}
	
	features_indices = {name: header.index(name) for name in features_names}
	
	X = []
	y = []
	
	nb_rows = len(data[houses_index])
	for i in range(nb_rows):
		house = data[houses_index][i]
		if house is None:
			continue
		
		features = []
		has_all_features = True
		for feature_name in features_names:
			feature_idx = features_indices[feature_name]
			value = data[feature_idx][i]
			if value is None:
				has_all_features = False
				break
			features.append(value)
		
		if has_all_features:
			X.append(features)
			y.append(house_to_label[house])
	
	m = len(X)
	print("Ending prepare_data")
	return X, y, m, houses

def _predict_multiclass_ovr(standardized_data: List[List[float]], thetas: List[List[float]]) -> List[int]:
	"""Argmax des scores sigmoïdes (one-vs-rest), cohérent avec logreg_predict."""
	m = len(standardized_data)
	n_features = len(standardized_data[0])
	y_pred: List[int] = []
	for i in range(m):
		best_k = 0
		best_s = -1.0
		for k, theta in enumerate(thetas):
			z = theta[0]
			for j in range(n_features):
				z += theta[j + 1] * standardized_data[i][j]
			s = utils.sigmoid(z)
			if s > best_s:
				best_s = s
				best_k = k
		y_pred.append(best_k)
	return y_pred

def train_model(X: List[List[float]], y: List[int], m: int, features_names: List[str], house_labels: List[str], model_path: str) -> None:
	"""
	Entraîne le modèle (à réimplémenter).

	À terme:
	- convertir y -> y_binary pour chaque maison k (one-vs-all)
	- entraîner 4 modèles binaires (gradient descent)
	- sauvegarder les poids pour logreg_predict
	"""
	print("Starting train_model")
	try:
		learning_rate = 0.01
		max_epochs = 1000
		epsilon = 1e-15 # 1e-15 = 0.000000000000001
		convergence_threshold = 1e-3 # 1e-3 = 0.001
		patience = 50
		standardized_data, standardization_params = utils.standardize_matrix(X)
		if standardized_data is None or standardization_params is None:
			raise Exception("Failed to standardize data")
		all_thetas = []
		for k in range(len(house_labels)):
			y_binary_k = utils.transform_ybinary(y, k)
			theta_k = train_model_binary(standardized_data, y_binary_k, learning_rate, max_epochs, epsilon, convergence_threshold, patience)
			all_thetas.append(theta_k)
		y_pred = _predict_multiclass_ovr(standardized_data, all_thetas)
		sklearn_acc = accuracy_score(y, y_pred)
		print("", flush=True)
		print("=" * 50, flush=True)
		print(f"Accuracy (scikit-learn) — jeu d'entraînement : {sklearn_acc:.4f}", flush=True)
		print("=" * 50, flush=True)
		print("", flush=True)
		if not utils.save_logreg_model_json(model_path, features_names, standardization_params, all_thetas, house_labels):
			raise Exception("Failed to save model JSON")
		print(f"Modèle écrit: {model_path}")
		print(f"all_thetas: {all_thetas}")
			 
		
	except Exception as e:
		print(f"Error: {e}")
		raise e
	print("Ending train_model")
	return None

def train_model_binary(standardized_data: List[List[float]], y_binary_k: List[int], learning_rate: float, max_epochs: int, epsilon: float, convergence_threshold: float, patience: int) -> List[float]:
	"""
	Entraîne le modèle binaire pour la maison k.
	"""
	print("Starting train_model_binary")
	theta = [0.0] * (len(standardized_data[0]) + 1)
	m = len(standardized_data)
	n_features = len(standardized_data[0])
	previous_cost = float('inf')
	print_every = 50
	for epoch in range(max_epochs):
		predictions = []
		for i in range(m):
			z = theta[0] * 1  # x₀ = 1 (biais)
			for j in range(n_features):
				z += theta[j+1] * standardized_data[i][j]
			h = utils.sigmoid(z)
			predictions.append(h)
		cost = 0.0
		for i in range(m):
			h = predictions[i]
			h = max(epsilon, min(1.0 - epsilon, h))
			if y_binary_k[i] == 1:
				cost += utils.ln(h)
			else:
				cost += utils.ln(1.0 - h)
		cost = -(1.0 / m) * cost
		if epoch > 0 and abs(previous_cost - cost) < convergence_threshold:
			break
		previous_cost = cost
		gradient = [0.0] * len(theta)
		for j in range(len(theta)):
			sum_error = 0.0
			for i in range(m):
				error = predictions[i] - y_binary_k[i]
				x_ij = 1.0 if j == 0 else standardized_data[i][j-1]
				sum_error += error * x_ij
			gradient[j] = (1.0 / m) * sum_error

		# correct = 0
		# for i in range(m):
		# 	pred_label = 1 if predictions[i] >= 0.5 else 0
		# 	if pred_label == y_binary_k[i]:
		# 		correct += 1

		# Prints utiles: coût + norme gradient + accuracy binaire (pas de spam)
		if epoch % print_every == 0:
			grad_l2 = (sum(g * g for g in gradient)) ** 0.5
			correct = 0
			for i in range(m):
				pred_label = 1 if predictions[i] >= 0.5 else 0
				if pred_label == y_binary_k[i]:
					correct += 1
			acc = correct / m
			os.system('clear')
			print(f"epoch: {epoch}")
			print(f"cost: {cost}")
			print(f"grad_l2: {grad_l2}")
			print(f"acc_binary: {acc}")

		for j in range(len(theta)):
			theta[j] = theta[j] - learning_rate * gradient[j]
	# print(f"Accuracy: {correct / m}")
	print("Ending train_model_binary")
	return theta


def main() -> bool:
	print("Starting main")
	try:
		if len(sys.argv) != 2:
			raise AttributeError("You must input 1 path")
		path = sys.argv[1]

		header, data, courses_names = extract_data(path)
		print(f"selected_features : {courses_names}")

		X, y, m, house_labels = prepare_data(header, data, courses_names)
		model_path = "model.json"
		train_model(X, y, m, courses_names, house_labels, model_path)
	except (Exception) as e:
		print(f"Error: {e}")
		return False
	return True


if __name__ == "__main__":
	main()