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
		
	except (Exception) as e:
		print(f"Error: {e}")
		return False
	return True



if __name__ == "__main__":
	logreg_train()