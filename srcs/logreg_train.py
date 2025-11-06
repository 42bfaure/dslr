import utils
import sys
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt

def create_houses_tab_scores(header: List[str], data: List[List], courses_names: List[str]) -> dict[str, dict[str, List[float]]]:
	"""Crée un dictionnaire organisant les scores par maison et par cours."""
	houses_index = header.index("Hogwarts House")
	courses_index = {course: header.index(course) for course in courses_names}
	houses = sorted({h for h in data[houses_index] if h is not None})

	houses_scores: dict[str, dict[str, List[float]]] = {
		house: {course: [] for course in courses_names} for house in houses
	}

	nb_rows = len(data[houses_index])
	for i in range(nb_rows):
		house = data[houses_index][i]
		if house is None or house not in houses_scores:
			continue
		for course, index in courses_index.items():
			score = data[index][i]
			if score is not None:
				houses_scores[house][course].append(score)
	return houses_scores

def logreg_train():
	try:
		if (len(sys.argv) != 2):
			raise AttributeError("You must input 1 path")
		header, data = utils.load_csv(sys.argv[1], columns=None, skip_header=True, 
							  return_header=True, auto_convert=True)
		if data is None:
			raise Exception("Failed to load data")
		numeric_names, numeric_data = utils.extract_numeric_columns(sys.argv[1], data, skip_header=True)
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
		houses_scores = create_houses_tab_scores(header, data, courses_names)
		homogeneity = utils.calculate_homogeneity(houses_scores)
		homogeneity_after_gap = utils.return_homogeneity_after_gap(homogeneity)
		print(f"\n📈 Homogénéité après gap (avec features sélectionnées):")
		print(f"{homogeneity_after_gap}")
		print(f"selected_features : {selected_features}")
		
		# Préparer les données pour l'entraînement
		# X = liste de listes : chaque ligne = un étudiant avec ses features
		# y = liste de labels : chaque élément = la maison de l'étudiant correspondant
		# m = nombre total d'étudiants
		X, y, m = prepare_training_data(header, data, courses_names)
		print(f"\n📊 Données préparées :")
		print(f"  - Nombre d'étudiants (m) : {m}")
		print(f"  - Nombre de features : {len(courses_names)}")
		print(f"  - Exemple X[0] (premier étudiant) : {X[0] if len(X) > 0 else 'N/A'}")
		print(f"  - Exemple y[0] (maison du premier étudiant) : {y[0] if len(y) > 0 else 'N/A'}")
		
		# Entraîner le modèle de régression logistique
		theta = logistic_regression(X, y, m, len(courses_names))
		print(f"theta : {theta}")
	except (Exception) as e:
		print(f"Error: {e}")
		return False
	return True

def prepare_training_data(header: List[str], data: List[List], features_names: List[str]) -> Tuple[List[List[float]], List[int], int]:
	"""
	Prépare les données pour l'entraînement en format X (features) et y (labels).
	
	Args:
		header: En-têtes du CSV
		data: Données brutes (List[List])
		features_names: Liste des noms des features sélectionnées
	
	Returns:
		Tuple (X, y, m) où:
		- X: Liste de listes, chaque ligne = features d'un étudiant
		- y: Liste de labels (0, 1, 2, 3 pour les 4 maisons)
		- m: Nombre total d'étudiants
	"""
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
		
		# Récupérer les features de cet étudiant
		features = []
		has_all_features = True
		for feature_name in features_names:
			feature_idx = features_indices[feature_name]
			value = data[feature_idx][i]
			if value is None:
				has_all_features = False
				break
			features.append(value)
		
		# Si toutes les features sont présentes, ajouter cet étudiant
		if has_all_features:
			X.append(features)
			y.append(house_to_label[house])
	
	m = len(X)  # Nombre total d'étudiants
	return X, y, m

def logistic_regression(X: List[List[float]], y: List[int], m: int, n_features: int) -> List[float]:
	"""
	Effectue une régression logistique pour prédire la maison d'un étudiant.
	
	Args:
		X: Liste de listes, chaque ligne = features d'un étudiant
		   Exemple : [[10, 8, 15], [12, 9, 16], ...]
		   - X[i] = features de l'étudiant i
		   - X[i][j] = note du cours j pour l'étudiant i
		y: Liste de labels (0, 1, 2, 3 pour les 4 maisons)
		   Exemple : [0, 0, 1, 2, 3, 0, ...]
		   - y[i] = maison de l'étudiant i
		m: Nombre total d'étudiants (m = len(X) = len(y))
		n_features: Nombre de features (cours) sélectionnés
	
	Returns:
		Liste des paramètres theta après entraînement
		Structure : [θ₀ (bias), θ₁, θ₂, ..., θₙ]
		- theta[0] = θ₀ (biais)
		- theta[j+1] = θⱼ (poids du cours j)
	"""
	# Initialisation
	theta = [0.0] * (n_features + 1)  # +1 pour le biais (θ₀)
	learning_rate = 0.01
	max_epochs = 10000
	convergence_threshold = 1e-6
	patience = 50
	
	best_cost = float('inf')
	best_theta = None
	no_improvement = 0
	previous_cost = float('inf')
	
	print(f"\n🚀 Démarrage de l'entraînement :")
	print(f"  - Paramètres initiaux : {theta}")
	print(f"  - Learning rate : {learning_rate}")
	print(f"  - Max epochs : {max_epochs}")
	print(f"  - Patience : {patience}")
	
	# Boucle d'entraînement
	for epoch in range(max_epochs):
		# TODO: Implémenter ici les étapes de l'entraînement
		# 1. Calculer les prédictions h_θ(x) pour tous les exemples
		# 2. Calculer le coût J(θ)
		# 3. Vérifier convergence et patience
		# 4. Calculer le gradient ∂J/∂θⱼ
		# 5. Mettre à jour theta : θⱼ = θⱼ - α × gradient
		
		# Placeholder pour l'instant
		if epoch == 0:
			print(f"\n⚠️  Entraînement non implémenté - TODO à compléter")
			break
	
	return theta


if __name__ == "__main__":
	logreg_train()