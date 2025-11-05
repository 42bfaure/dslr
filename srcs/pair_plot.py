import utils
import sys
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt

def pair_plot():
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
		# Filtrer les None de toutes les colonnes
		clean_numeric_data = utils.none_filter(numeric_data)
		n_courses = len(numeric_names)
		fig, axes = plt.subplots(n_courses, n_courses, figsize=(20, 20))
		
		for i in range(n_courses):
			for j in range(n_courses):
				ax = axes[i, j]
				
				if i == j:
					# Diagonale : histogramme
					ax.hist(clean_numeric_data[i], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
					if j == 0:
						ax.set_ylabel(numeric_names[i], fontsize=8)
				else:
					# Hors diagonale : scatter plot
					# Filtrer les paires pour avoir les mêmes indices
					paired_data_x = []
					paired_data_y = []
					for k in range(min(len(numeric_data[j]), len(numeric_data[i]))):
						if numeric_data[j][k] is not None and numeric_data[i][k] is not None:
							paired_data_x.append(numeric_data[j][k])
							paired_data_y.append(numeric_data[i][k])
					
					ax.scatter(paired_data_x, paired_data_y, color='blue', alpha=0.3, s=1)
				
				# Labels seulement sur les bords
				if i == n_courses - 1:
					ax.set_xlabel(numeric_names[j], fontsize=8, rotation=45, ha='right')
				else:
					ax.set_xticklabels([])
				
				if j == 0:
					ax.set_ylabel(numeric_names[i], fontsize=8)
				else:
					ax.set_yticklabels([])
				
				# Réduire la taille des ticks
				ax.tick_params(labelsize=6)
		plt.tight_layout()
		plt.savefig("pair_plot.png", dpi=300, bbox_inches='tight')
		plt.show()
		return True
	except (Exception) as e:
		print(f"Error: {e}")
		return False

if __name__ == "__main__":
	pair_plot()