import utils
import sys
from typing import Union, List, Optional

def describe():
	try:
		if (len(sys.argv) != 2):
			raise AttributeError("You must input 1 path")
		
		# Charger les données avec skip_header=True pour exclure l'en-tête
		data = utils.load_csv(sys.argv[1], columns=None, skip_header=True, 
		                      return_header=False, auto_convert=True)
		
		if data is None:
			raise Exception("Failed to load data")
		
		# Extraire uniquement les colonnes numériques
		numeric_result = utils.extract_numeric_columns(sys.argv[1], data, skip_header=True)
		if numeric_result is None:
			raise Exception("No numeric data found")
		
		numeric_names, numeric_data = numeric_result
		
		# Filtrer pour exclure la colonne "Index" si elle existe
		filtered_names = []
		filtered_data = []
		for i, column_name in enumerate(numeric_names):
			if column_name.lower() != 'index' and i < len(numeric_data):
				filtered_names.append(column_name)
				filtered_data.append(numeric_data[i])
		
		# Calculer les statistiques pour toutes les colonnes (sauf Index)
		all_stats = {}
		for i, column_name in enumerate(filtered_names):
			if i < len(filtered_data):
				stats = describe_statistics(filtered_data[i])
				if stats:
					all_stats[column_name] = stats
		
		if not all_stats:
			print("No statistics to display")
			return None
		
		# Afficher sous forme de tableau
		print_statistics_table(all_stats, filtered_names)
		
	except (Exception) as e:
		print(f"Error: {e}")
		return None

def print_statistics_table(all_stats: dict, column_names: List[str]):
	"""
	Affiche les statistiques sous forme de tableau formaté.
	
	Args:
		all_stats: Dictionnaire {column_name: {stat_name: value}}
		column_names: Liste des noms de colonnes dans l'ordre
	"""
	# Métriques à afficher dans l'ordre
	metrics = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
	metric_labels = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
	
	# Largeur des colonnes
	col_width = 15
	label_width = 10
	
	# Ligne d'en-tête avec les noms de colonnes
	header = ' ' * label_width
	for col_name in column_names:
		if col_name in all_stats:
			# Tronquer si le nom est trop long
			display_name = col_name[:col_width-1] if len(col_name) >= col_width else col_name
			header += f"{display_name:>{col_width}}"
	print(header)
	
	# Lignes de statistiques
	for i, metric in enumerate(metrics):
		line = f"{metric_labels[i]:<{label_width}}"
		for col_name in column_names:
			if col_name in all_stats and metric in all_stats[col_name]:
				value = all_stats[col_name][metric]
				if isinstance(value, (int, float)):
					line += f"{value:>{col_width}.6f}"
				else:
					line += f"{str(value):>{col_width}}"
			else:
				line += ' ' * col_width
		print(line)

def describe_statistics(data: List[Union[int, float]]) -> Optional[dict]:
	"""
	Calcule toutes les statistiques descriptives d'une liste.
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Dictionnaire avec toutes les statistiques ou None en cas d'erreur
	"""
	try:
		if len(data) == 0:
			return None
		
		# Filtrer les valeurs None
		clean_data = [x for x in data if x is not None]
		
		if len(clean_data) == 0:
			return None
		
		min_val, max_val = utils.get_min_max(clean_data)
		q1, q2, q3 = utils.quartiles(clean_data)
		
		return {
			'count': len(clean_data),
			'mean': utils.mean(clean_data),
			'std': utils.std(clean_data),
			'min': min_val,
			'25%': q1,
			'50%': q2,  # médiane
			'75%': q3,
			'max': max_val,
			'range': max_val - min_val or 0,
			'iqr': q3 - q1  # Inter-Quartile Range
		}
		
	except Exception as e:
		print(f"Erreur lors du calcul des statistiques: {e}")
		return None

if __name__ == "__main__":
    describe()