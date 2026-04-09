import utils
import sys
from typing import Union, List, Optional, Tuple

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
		print("\nℹ️  Entraînement désactivé (à réimplémenter).")
		print("    Prochaine étape: coder l'entraînement one-vs-all à partir de X, y.")
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
	return X, y, m


if __name__ == "__main__":
	logreg_train()