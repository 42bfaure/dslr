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
		selected_features_data = get_selected_features_data(houses_scores, courses_names)
		
		theta = logistic_regression(selected_features_data, courses_names)
		print(f"theta : {theta}")
	except (Exception) as e:
		print(f"Error: {e}")
		return False
	return True

def get_selected_features_data(houses_scores: dict[str, dict[str, List[float]]], selected_features: List[str]) -> dict[str, dict[str, List[float]]]:
	"""
	Retourne les données des features sélectionnées.
	
	Args:
		houses_scores: Dictionnaire des scores par maison et par cours
		               Structure: {maison: {cours: [valeurs]}}
		selected_features: Liste des features sélectionnées (noms des cours)
	
	Returns:
		Dictionnaire avec la même structure mais filtré pour les features sélectionnées
		Structure: {maison: {cours: [valeurs]}}
		Exemple: {
			"Gryffindor": {
				"Arithmancy": [10, 8, 12, ...],
				"Astronomy": [15, 14, 16, ...]
			},
			"Hufflepuff": {
				"Arithmancy": [9, 7, 11, ...],
				"Astronomy": [13, 12, 14, ...]
			}
		}
	"""
	return {
		house: {
			feature: houses_scores[house][feature] 
			for feature in selected_features 
			if feature in houses_scores[house]
		}
		for house in houses_scores
	}

def logistic_regression(selected_features_data: dict[str, dict[str, List[float]]], features_names: List[str]) -> List[float]:
	"""
	Effectue une régression logistique pour prédire la maison d'un étudiant.
	
	Args:
		selected_features_data: Dictionnaire des données filtrées
		                        Structure: {maison: {cours: [valeurs]}}
		features_names: Liste des noms des features sélectionnées
	
	Returns:
		Liste des paramètres theta après entraînement
	"""
	theta = [0.0] * (len(features_names) + 1)  # +1 pour le biais
	print(f"theta initialisé : {theta}")
	learning_rate = 0.01
	max_epochs = 10000
	convergence_threshold = 1e-6
	patience = 50

	
	return theta


if __name__ == "__main__":
	logreg_train()