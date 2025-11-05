import utils
import sys
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt

def scatter_plot():
	try:
		if (len(sys.argv) != 2):
			raise AttributeError("You must input 1 path")
		header, data = utils.load_csv(sys.argv[1], columns=None, skip_header=True, 
		                      return_header=True, auto_convert=True)
		if data is None:
			raise Exception("Failed to load data")
		numeric_result = utils.extract_numeric_columns(sys.argv[1], data, skip_header=True)
		if numeric_result is None:
			raise Exception("No numeric data found")
		numeric_names, numeric_data = numeric_result
		
		# Calculer toutes les corrélations
		max_correlation = 0
		best_pair = None
		
		for i, course1 in enumerate(numeric_names):
			for j, course2 in enumerate(numeric_names):
				if i < j:  # Éviter les doublons et la diagonale
					# Filtrer les données pour enlever les None
					clean_data1, clean_data2 = filter_paired_data(numeric_data[i], numeric_data[j])
					
					if len(clean_data1) > 0:
						correlation = utils.correlation_coefficient(clean_data1, clean_data2)
						print(f"Correlation between {course1} and {course2}: {correlation:.4f}")
						
						# Garder la paire avec la plus forte corrélation POSITIVE
						if abs(correlation) > abs(max_correlation):
							max_correlation = correlation
							best_pair = (i, j, course1, course2, clean_data1, clean_data2)
		
		if best_pair is not None:
			i, j, course1, course2, data1, data2 = best_pair
			print(f"\n🏆 Paire la plus corrélée: {course1} et {course2} (r = {max_correlation:.4f})")
			utils.plot_scatter(data1, data2, f"scatter_{course1}_vs_{course2}.png", x_label=course1, y_label=course2, title=f"{course1} vs {course2}")
		
		return True
	except (Exception) as e:
		print(f"Error: {e}")

def filter_paired_data(data1: List, data2: List) -> Tuple[List[float], List[float]]:
	"""Filtre les paires de données pour enlever les valeurs None"""
	clean_data1 = []
	clean_data2 = []
	
	for i in range(len(data1)):
		if i < len(data2) and data1[i] is not None and data2[i] is not None:
			clean_data1.append(data1[i])
			clean_data2.append(data2[i])
	
	return clean_data1, clean_data2

if __name__ == "__main__":
	scatter_plot()
