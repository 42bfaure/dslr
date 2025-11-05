import csv
from typing import Union, List, Tuple, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt

def load_csv(path: str, columns: Optional[List[int]] = None, skip_header: bool = True, 
			 convert_type: Optional[type] = float, auto_convert: bool = False, 
			 parse_dates: bool = True, return_header: bool = False) -> Optional[Union[List[List], Tuple[List[str], List[List]]]]:
	"""
	Charge des données depuis un fichier CSV.
	
	Args:
		path: Chemin vers le fichier CSV
		columns: Liste des indices de colonnes à charger (None = toutes les colonnes)
		skip_header: Si True, ignore la première ligne (sauf si return_header=True)
		convert_type: Type de conversion (float, int, str, None pour aucune conversion)
		auto_convert: Si True, tente une conversion automatique intelligente
		parse_dates: Si True avec auto_convert, tente de parser les dates
		return_header: Si True, retourne (header, data) au lieu de juste data
	
	Returns:
		Liste de listes contenant les données de chaque colonne, ou None en cas d'erreur
		Si return_header=True: tuple (header, data)
	"""
	try:
		if not isinstance(path, str):
			raise TypeError("Le chemin doit être une chaîne de caractères")
		
		header = None
		data = []
		
		with open(path, 'r') as file:
			csv_reader = csv.reader(file)
			
			# Logique de gestion de l'en-tête
			if return_header:
				# Si on veut retourner le header, on doit le lire
				header_row = next(csv_reader)
				if columns is not None:
					header = [header_row[i] for i in columns]
				else:
					header = header_row
			elif skip_header:
				# Si on veut juste skip sans retourner, on saute la ligne
				next(csv_reader)
			# Sinon (skip_header=False et return_header=False), on ne touche pas à la première ligne
			
			for row in csv_reader:
				if columns is None:
					selected_values = row
				else:
					selected_values = [row[i] for i in columns]
				
				# Conversion selon les paramètres
				if auto_convert:
					converted_row = [_auto_convert_value(val, parse_dates) for val in selected_values]
				elif convert_type is not None:
					converted_row = [convert_type(val) for val in selected_values]
				else:
					converted_row = selected_values
				
				data.append(converted_row)
		
		# Transpose pour avoir une liste par colonne
		if data:
			transposed_data = [list(col) for col in zip(*data)]
		else:
			transposed_data = []
		
		if return_header:
			return (header, transposed_data)
		return transposed_data
		
	except Exception as e:
		print(f"Erreur lors du chargement: {e}")
		return None

def _auto_convert_value(value: str, parse_dates: bool = True) -> Any:
	"""
	Convertit automatiquement une valeur string vers le type le plus approprié.
	
	Args:
		value: Valeur à convertir
		parse_dates: Si True, tente de parser les dates
	
	Returns:
		Valeur convertie (int, float, datetime, str ou None)
	"""
	value = value.strip()
	
	# Valeur vide
	if value == '' or value.lower() in ['nan', 'null', 'none', 'n/a']:
		return None
	
	# Tentative de conversion en int
	try:
		if '.' not in value and 'e' not in value.lower() and '-' not in value:
			return int(value)
	except ValueError:
		pass
	
	# Tentative de conversion en float
	try:
		float_val = float(value)
		return float_val
	except ValueError:
		pass
	
	# Tentative de conversion en date
	if parse_dates:
		date_formats = [
			'%Y-%m-%d',           # 2000-03-30
			'%d/%m/%Y',           # 30/03/2000
			'%m/%d/%Y',           # 03/30/2000
			'%Y-%m-%d %H:%M:%S',  # 2000-03-30 12:30:45
			'%d-%m-%Y',           # 30-03-2000
			'%Y/%m/%d',           # 2000/03/30
		]
		
		for date_format in date_formats:
			try:
				return datetime.strptime(value, date_format)
			except ValueError:
				continue
	
	# Sinon, reste en string
	return value

def analyze_csv_types(path: str, sample_size: int = 100) -> Optional[List[dict]]:
	"""
	Analyse les types de données de chaque colonne d'un CSV.
	
	Args:
		path: Chemin vers le fichier CSV
		sample_size: Nombre de lignes à analyser (None = toutes les lignes)
	
	Returns:
		Liste de dictionnaires avec les infos de chaque colonne:
		{'name': str, 'type': str, 'null_count': int, 'sample': Any}
	"""
	try:
		with open(path, 'r') as file:
			csv_reader = csv.reader(file)
			headers = next(csv_reader)
			
			# Initialiser les compteurs
			column_info = []
			for header in headers:
				column_info.append({
					'name': header,
					'types': {'int': 0, 'float': 0, 'date': 0, 'str': 0, 'null': 0},
					'samples': []
				})
			
			# Analyser les lignes
			for i, row in enumerate(csv_reader):
				if sample_size is not None and i >= sample_size:
					break
				
				for col_idx, value in enumerate(row):
					converted = _auto_convert_value(value)
					
					# Garder quelques échantillons
					if len(column_info[col_idx]['samples']) < 3:
						column_info[col_idx]['samples'].append(value)
					
					# Compter les types
					if converted is None:
						column_info[col_idx]['types']['null'] += 1
					elif isinstance(converted, int):
						column_info[col_idx]['types']['int'] += 1
					elif isinstance(converted, float):
						column_info[col_idx]['types']['float'] += 1
					elif isinstance(converted, datetime):
						column_info[col_idx]['types']['date'] += 1
					else:
						column_info[col_idx]['types']['str'] += 1
			
			# Déterminer le type dominant
			results = []
			for info in column_info:
				types = info['types']
				total = sum(types.values())
				
				# Trouver le type le plus fréquent
				dominant_type = max(types.items(), key=lambda x: x[1])[0]
				
				results.append({
					'name': info['name'],
					'dominant_type': dominant_type,
					'null_count': types['null'],
					'null_percentage': (types['null'] / total * 100) if total > 0 else 0,
					'samples': info['samples'][:3],
					'distribution': types
				})
			
			return results
			
	except Exception as e:
		print(f"Erreur lors de l'analyse: {e}")
		return None

def get_numeric_columns(path: str, skip_header: bool = True) -> Optional[Tuple[List[str], List[int]]]:
	"""
	Identifie les colonnes numériques d'un CSV.
	
	Args:
		path: Chemin vers le fichier CSV
		skip_header: Si True, ignore la première ligne
	
	Returns:
		Tuple (noms_colonnes, indices_colonnes) des colonnes numériques
	"""
	try:
		analysis = analyze_csv_types(path)
		if analysis is None:
			return None
		
		numeric_names = []
		numeric_indices = []
		
		for idx, col in enumerate(analysis):
			if col['dominant_type'] in ['int', 'float']:
				numeric_names.append(col['name'])
				numeric_indices.append(idx)
		
		return (numeric_names, numeric_indices)
	except Exception as e:
		print(f"Erreur lors de l'identification des colonnes numériques: {e}")
		return None

def extract_numeric_columns(path: str, data: Union[List[List], Tuple[List[str], List[List]]], 
							skip_header: bool = True) -> Optional[Tuple[List[str], List[List]]]:
	"""
	Extrait uniquement les colonnes numériques des données chargées.
	
	Args:
		path: Chemin vers le fichier CSV (utilisé pour analyser les types)
		data: Données chargées avec load_csv (avec ou sans header)
		skip_header: Si True, considère que le CSV a un en-tête
	
	Returns:
		Tuple (noms_colonnes_numériques, données_colonnes_numériques) ou None en cas d'erreur
	"""
	try:
		# Obtenir les indices et noms des colonnes numériques
		result = get_numeric_columns(path, skip_header)
		if result is None or data is None:
			return None
		
		numeric_names, numeric_indices = result
		
		# Déterminer si data contient un header ou non
		if isinstance(data, tuple) and len(data) == 2:
			# data est (header, columns_data)
			header, columns_data = data
		else:
			# data est juste columns_data
			columns_data = data
		
		# Filtrer pour ne garder que les colonnes numériques
		numeric_data = []
		for idx in numeric_indices:
			if idx < len(columns_data):
				numeric_data.append(columns_data[idx])
		
		return (numeric_names, numeric_data)
		
	except Exception as e:
		print(f"Erreur lors de l'extraction des colonnes numériques: {e}")
		return None

def none_filter(data: List[List]) -> List[List]:
	"""
	Filtre les valeurs None d'une liste de listes.
	
	Args:
		data: Liste de listes
	
	Returns:
		Liste de listes sans les valeurs None
	"""
	clean_data = []
	for column in data:
		clean_column = [x for x in column if x is not None]
		clean_data.append(clean_column)
	return clean_data

def count_missing_values(path: str, skip_header: bool = True) -> Optional[dict]:
	"""
	Compte les cellules vides et pleines pour chaque colonne d'un CSV.
	
	Args:
		path: Chemin vers le fichier CSV
		skip_header: Si True, ignore la première ligne
	
	Returns:
		Dictionnaire avec les statistiques de valeurs manquantes:
		{
			'columns': [{'name': str, 'empty': int, 'filled': int, 'total': int, 
			             'empty_percentage': float, 'filled_percentage': float}],
			'global': {'empty': int, 'filled': int, 'total': int, 
			           'empty_percentage': float, 'filled_percentage': float}
		}
	"""
	try:
		if not isinstance(path, str):
			raise TypeError("Le chemin doit être une chaîne de caractères")
		
		with open(path, 'r') as file:
			csv_reader = csv.reader(file)
			
			# Lire l'en-tête
			headers = next(csv_reader) if skip_header else None
			
			# Initialiser les compteurs par colonne
			column_stats = []
			for i, header in enumerate(headers if headers else []):
				column_stats.append({
					'name': header,
					'empty': 0,
					'filled': 0
				})
			
			# Si pas d'en-tête, on initialise avec la première ligne
			if not headers:
				first_row = next(csv_reader)
				for i in range(len(first_row)):
					column_stats.append({
						'name': f'Column_{i}',
						'empty': 0,
						'filled': 0
					})
				# Traiter la première ligne
				for col_idx, value in enumerate(first_row):
					if value.strip() == '' or value.strip().lower() in ['nan', 'null', 'none', 'n/a']:
						column_stats[col_idx]['empty'] += 1
					else:
						column_stats[col_idx]['filled'] += 1
			
			# Parcourir toutes les lignes
			for row in csv_reader:
				for col_idx, value in enumerate(row):
					if col_idx < len(column_stats):
						if value.strip() == '' or value.strip().lower() in ['nan', 'null', 'none', 'n/a']:
							column_stats[col_idx]['empty'] += 1
						else:
							column_stats[col_idx]['filled'] += 1
		
		# Calculer les statistiques par colonne
		columns_results = []
		total_empty = 0
		total_filled = 0
		
		for col in column_stats:
			total = col['empty'] + col['filled']
			empty_pct = (col['empty'] / total * 100) if total > 0 else 0
			filled_pct = (col['filled'] / total * 100) if total > 0 else 0
			
			columns_results.append({
				'name': col['name'],
				'empty': col['empty'],
				'filled': col['filled'],
				'total': total,
				'empty_percentage': round(empty_pct, 2),
				'filled_percentage': round(filled_pct, 2)
			})
			
			total_empty += col['empty']
			total_filled += col['filled']
		
		# Statistiques globales
		total_cells = total_empty + total_filled
		global_empty_pct = (total_empty / total_cells * 100) if total_cells > 0 else 0
		global_filled_pct = (total_filled / total_cells * 100) if total_cells > 0 else 0
		
		return {
			'columns': columns_results,
			'global': {
				'empty': total_empty,
				'filled': total_filled,
				'total': total_cells,
				'empty_percentage': round(global_empty_pct, 2),
				'filled_percentage': round(global_filled_pct, 2)
			}
		}
		
	except Exception as e:
		print(f"Erreur lors du comptage des valeurs manquantes: {e}")
		return None

def print_missing_values_report(path: str, skip_header: bool = True, show_all_columns: bool = False) -> bool:
	"""
	Affiche un rapport détaillé des valeurs manquantes.
	
	Args:
		path: Chemin vers le fichier CSV
		skip_header: Si True, ignore la première ligne
		show_all_columns: Si True, affiche toutes les colonnes, sinon seulement celles avec des valeurs manquantes
	
	Returns:
		True si succès, False sinon
	"""
	try:
		stats = count_missing_values(path, skip_header)
		if stats is None:
			return False
		
		print("\n" + "="*80)
		print(f"RAPPORT DES VALEURS MANQUANTES - {path}")
		print("="*80)
		
		print("\n📊 STATISTIQUES GLOBALES:")
		print(f"  Total de cellules      : {stats['global']['total']:,}")
		print(f"  Cellules pleines       : {stats['global']['filled']:,} ({stats['global']['filled_percentage']:.2f}%)")
		print(f"  Cellules vides         : {stats['global']['empty']:,} ({stats['global']['empty_percentage']:.2f}%)")
		
		print("\n📋 STATISTIQUES PAR COLONNE:")
		print(f"{'Colonne':<30} {'Vides':<12} {'Pleines':<12} {'% Vide':<10} {'% Plein':<10}")
		print("-"*80)
		
		for col in stats['columns']:
			if show_all_columns or col['empty'] > 0:
				print(f"{col['name']:<30} {col['empty']:<12} {col['filled']:<12} "
				      f"{col['empty_percentage']:<10.2f} {col['filled_percentage']:<10.2f}")
		
		if not show_all_columns:
			clean_cols = [col for col in stats['columns'] if col['empty'] == 0]
			if clean_cols:
				print(f"\n✅ {len(clean_cols)} colonne(s) sans valeurs manquantes (non affichées)")
		
		print("="*80 + "\n")
		return True
		
	except Exception as e:
		print(f"Erreur lors de l'affichage du rapport: {e}")
		return False

def save_model_params(params: List[Union[int, float]], path: str) -> bool:
	"""
	Sauvegarde les paramètres d'un modèle dans un fichier.
	
	Args:
		params: Liste des paramètres à sauvegarder
		path: Chemin du fichier de sortie
	
	Returns:
		True si succès, False sinon
	"""
	try:
		if not all(isinstance(p, (int, float)) for p in params):
			raise TypeError("Tous les paramètres doivent être des nombres")
		if not isinstance(path, str):
			raise TypeError("Le chemin doit être une chaîne de caractères")
		
		with open(path, "w") as f:
			f.write(" ".join(map(str, params)))
		return True
	except Exception as e:
		print(f"Erreur lors de la sauvegarde: {e}")
		return False

def load_model_params(path: str, expected_count: Optional[int] = None) -> Optional[List[float]]:
	"""
	Charge les paramètres d'un modèle depuis un fichier.
	
	Args:
		path: Chemin vers le fichier de paramètres
		expected_count: Nombre de paramètres attendus (None = pas de vérification)
	
	Returns:
		Liste des paramètres chargés, ou None en cas d'erreur
	"""
	try:
		if not isinstance(path, str):
			raise TypeError("Le chemin doit être une chaîne de caractères")
		
		with open(path, 'r') as f:
			params = list(map(float, f.read().split()))
		
		if expected_count is not None and len(params) != expected_count:
			raise ValueError(f"Nombre de paramètres incorrect: {len(params)} au lieu de {expected_count}")
		
		return params
	except Exception as e:
		print(f"Erreur lors du chargement: {e}")
		return None

def ft_min(data: List[Union[int, float]]) -> Optional[float]:
	"""
	Trouve la valeur minimale d'une liste sans utiliser min().
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Valeur minimale ou None en cas d'erreur
	"""
	try:
		if len(data) == 0:
			raise ValueError("La liste ne peut pas être vide")
		
		min_val = data[0]
		for val in data[1:]:
			if val < min_val:
				min_val = val
		
		return min_val
	except Exception as e:
		print(f"Erreur dans ft_min: {e}")
		return None

def ft_max(data: List[Union[int, float]]) -> Optional[float]:
	"""
	Trouve la valeur maximale d'une liste sans utiliser max().
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Valeur maximale ou None en cas d'erreur
	"""
	try:
		if len(data) == 0:
			raise ValueError("La liste ne peut pas être vide")
		
		max_val = data[0]
		for val in data[1:]:
			if val > max_val:
				max_val = val
		
		return max_val
	except Exception as e:
		print(f"Erreur dans ft_max: {e}")
		return None

def get_min_max(data: List[Union[int, float]]) -> Optional[Tuple[float, float]]:
	"""
	Calcule les valeurs min et max d'une liste.
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Tuple (min, max) ou None en cas d'erreur
	"""
	try:
		if not all(isinstance(x, (int, float)) for x in data):
			raise TypeError("Toutes les valeurs doivent être numériques")
		if len(data) == 0:
			raise ValueError("La liste ne peut pas être vide")
		return (ft_min(data), ft_max(data))
	except Exception as e:
		print(f"Erreur: {e}")
		return None

def normalize_min_max(data: List[Union[int, float]], min_val: Optional[float] = None, max_val: Optional[float] = None) -> Optional[List[float]]:
	"""
	Normalise les données entre 0 et 1 avec la méthode min-max.
	
	Args:
		data: Liste de valeurs à normaliser
		min_val: Valeur minimale (calculée si None)
		max_val: Valeur maximale (calculée si None)
	
	Returns:
		Liste des valeurs normalisées, ou None en cas d'erreur
	"""
	try:
		if min_val is None or max_val is None:
			result = get_min_max(data)
			if result is None:
				return None
			min_val, max_val = result
		
		if max_val == min_val:
			return [0.5] * len(data)
		
		return [(x - min_val) / (max_val - min_val) for x in data]
	except Exception as e:
		print(f"Erreur lors de la normalisation: {e}")
		return None

def standardize(data: List[Union[int, float]]) -> Optional[Tuple[List[float], float, float]]:
	"""
	Standardise les données (moyenne=0, écart-type=1).
	
	Args:
		data: Liste de valeurs à standardiser
	
	Returns:
		Tuple (données_standardisées, moyenne, écart_type) ou None en cas d'erreur
	"""
	try:
		if len(data) == 0:
			raise ValueError("La liste ne peut pas être vide")
		
		mean = sum(data) / len(data)
		variance = sum((x - mean) ** 2 for x in data) / len(data)
		std = variance ** 0.5
		
		if std == 0:
			return [0.0] * len(data), mean, std
		
		standardized = [(x - mean) / std for x in data]
		return standardized, mean, std
	except Exception as e:
		print(f"Erreur lors de la standardisation: {e}")
		return None

def plot_scatter(x_data: List[Union[int, float]], y_data: List[Union[int, float]], 
				 output_path: str, x_label: str = "X", y_label: str = "Y", 
				 title: str = "Nuage de points", regression_line: Optional[Tuple[float, float]] = None,
				 norm_params: Optional[Tuple[float, float]] = None, figsize: Tuple[int, int] = (12, 8)) -> bool:
	"""
	Crée un graphique de dispersion avec option de ligne de régression.
	
	Args:
		x_data: Données de l'axe X
		y_data: Données de l'axe Y
		output_path: Chemin de sauvegarde du graphique
		x_label: Label de l'axe X
		y_label: Label de l'axe Y
		title: Titre du graphique
		regression_line: Tuple (intercept, slope) pour tracer une ligne de régression
		norm_params: Tuple (min_val, max_val) si les données X sont normalisées
		figsize: Taille de la figure
	
	Returns:
		True si succès, False sinon
	"""
	try:
		plt.figure(figsize=figsize)
		plt.scatter(x_data, y_data, color='blue', alpha=0.6, label='Données', s=50)

		if regression_line is not None:
			intercept, slope = regression_line
			x_range = [ft_min(x_data), ft_max(x_data)]
			
			if norm_params is not None:
				min_val, max_val = norm_params
				x_normalized = normalize_min_max(x_range, min_val, max_val)
				y_pred = [linear_prediction(x, intercept, slope) for x in x_normalized]
			else:
				y_pred = [linear_prediction(x, intercept, slope) for x in x_range]
			
			plt.plot(x_range, y_pred, color='red', linewidth=2, label='Régression')
			
			r_squared = calculate_r2(x_data, y_data, intercept, slope, norm_params)
			plt.title(f'{title} - R²: {r_squared:.4f}')
		else:
			plt.title(title)
		
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.grid(True, linestyle='--', alpha=0.7)
		plt.legend()
		plt.tight_layout()
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		plt.show()
		return True
		
	except Exception as e:
		print(f"Erreur lors du tracé: {e}")
		return False

def calculate_r2(x_data: List[Union[int, float]], y_data: List[Union[int, float]], 
				 intercept: float, slope: float, norm_params: Optional[Tuple[float, float]] = None) -> float:
	"""
	Calcule le coefficient de détermination R² pour une régression linéaire.
	
	Args:
		x_data: Données d'entrée
		y_data: Données de sortie réelles
		intercept: Ordonnée à l'origine du modèle
		slope: Pente du modèle
		norm_params: Tuple (min_val, max_val) si les données sont normalisées
	
	Returns:
		Valeur R² (entre 0 et 1, 1 étant parfait)
	"""
	try:
		if norm_params is not None:
			min_val, max_val = norm_params
			x_normalized = normalize_min_max(x_data, min_val, max_val)
			predictions = [linear_prediction(x, intercept, slope) for x in x_normalized]
		else:
			predictions = [linear_prediction(x, intercept, slope) for x in x_data]
		
		mean_y = sum(y_data) / len(y_data)
		ss_tot = sum((y - mean_y) ** 2 for y in y_data)
		ss_res = sum((y - pred) ** 2 for y, pred in zip(y_data, predictions))
		
		return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
		
	except Exception as e:
		print(f"Erreur lors du calcul de R²: {e}")
		return 0.0

def calculate_mse(y_true: List[Union[int, float]], y_pred: List[Union[int, float]]) -> Optional[float]:
	"""
	Calcule l'erreur quadratique moyenne (Mean Squared Error).
	
	Args:
		y_true: Valeurs réelles
		y_pred: Valeurs prédites
	
	Returns:
		MSE ou None en cas d'erreur
	"""
	try:
		if len(y_true) != len(y_pred):
			raise ValueError("Les listes doivent avoir la même longueur")
		return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
	except Exception as e:
		print(f"Erreur lors du calcul de MSE: {e}")
		return None

def calculate_mae(y_true: List[Union[int, float]], y_pred: List[Union[int, float]]) -> Optional[float]:
	"""
	Calcule l'erreur absolue moyenne (Mean Absolute Error).
	
	Args:
		y_true: Valeurs réelles
		y_pred: Valeurs prédites
	
	Returns:
		MAE ou None en cas d'erreur
	"""
	try:
		if len(y_true) != len(y_pred):
			raise ValueError("Les listes doivent avoir la même longueur")
		return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
	except Exception as e:
		print(f"Erreur lors du calcul de MAE: {e}")
		return None

def linear_prediction(x: Union[int, float], intercept: float, slope: float) -> float:
	"""
	Effectue une prédiction avec un modèle linéaire: y = intercept + slope * x
	
	Args:
		x: Valeur d'entrée
		intercept: Ordonnée à l'origine
		slope: Pente
	
	Returns:
		Valeur prédite
	"""
	return intercept + (slope * x)


def count(data: List[List]) -> Tuple[int, List[int]]:
	"""
	Compte le nombre de lignes pour chaque colonne dans une liste.
	
	Args:
		data: Liste de listes
	
	Returns:
		Tuple (nombre de colonnes, liste (nombre de lignes pour chaque colonne))
	"""
	return len(data), [len(col) for col in data]

def mean(data: List[Union[int, float]]) -> Optional[float]:
	"""
	Calcule la moyenne d'une liste.
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Moyenne ou None en cas d'erreur
	"""
	return sum(data) / len(data)

def std(data: List[Union[int, float]]) -> Optional[float]:
	"""
	Calcule l'écart-type d'une liste.
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Écart-type ou None en cas d'erreur
	"""
	return (sum((x - mean(data)) ** 2 for x in data) / len(data)) ** 0.5

def median(data: List[Union[int, float]]) -> Optional[float]:
	"""
	Calcule la médiane d'une liste (50e percentile).
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Médiane ou None en cas d'erreur
	"""
	try:
		if len(data) == 0:
			return None
		
		sorted_data = sorted(data)
		n = len(sorted_data)
		
		if n % 2 == 0:
			# Si nombre pair, moyenne des deux valeurs centrales
			return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
		else:
			# Si nombre impair, valeur centrale
			return sorted_data[n // 2]
	except Exception as e:
		print(f"Erreur lors du calcul de la médiane: {e}")
		return None

def percentile(data: List[Union[int, float]], p: float) -> Optional[float]:
	"""
	Calcule le percentile d'une liste.
	
	Args:
		data: Liste de valeurs numériques
		p: Percentile à calculer (entre 0 et 100)
	
	Returns:
		Valeur du percentile ou None en cas d'erreur
	"""
	try:
		if len(data) == 0:
			return None
		if p < 0 or p > 100:
			raise ValueError("Le percentile doit être entre 0 et 100")
		
		sorted_data = sorted(data)
		n = len(sorted_data)
		
		# Calcul de l'index (méthode linéaire)
		index = (p / 100) * (n - 1)
		
		# Si l'index est entier, retourner directement la valeur
		if index.is_integer():
			return sorted_data[int(index)]
		
		# Sinon, interpolation linéaire entre deux valeurs
		lower_index = int(index)
		upper_index = lower_index + 1
		weight = index - lower_index
		
		return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
		
	except Exception as e:
		print(f"Erreur lors du calcul du percentile: {e}")
		return None

def quartiles(data: List[Union[int, float]]) -> Optional[Tuple[float, float, float]]:
	"""
	Calcule les quartiles (Q1, Q2/médiane, Q3) d'une liste.
	
	Args:
		data: Liste de valeurs numériques
	
	Returns:
		Tuple (Q1, Q2, Q3) ou None en cas d'erreur
		- Q1 (25%): 25% des données sont en dessous
		- Q2 (50%): Médiane
		- Q3 (75%): 75% des données sont en dessous
	"""
	try:
		q1 = percentile(data, 25)
		q2 = percentile(data, 50)  # ou median(data)
		q3 = percentile(data, 75)
		
		if q1 is None or q2 is None or q3 is None:
			return None
		
		return (q1, q2, q3)
		
	except Exception as e:
		print(f"Erreur lors du calcul des quartiles: {e}")
		return None

def correlation_coefficient(x: List[Union[int, float]], y: List[Union[int, float]]) -> Optional[float]:
	"""
	Calcule le coefficient de corrélation de Pearson entre deux listes.
	
	Args:
		x: Liste de valeurs numériques
		y: Liste de valeurs numériques
	
	Returns:
		Coefficient de corrélation ou None en cas d'erreur
	"""

	n = len(x)
	mean_x = sum(x) / n
	mean_y = sum(y) / n

	covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
	
	std_x = std(x)
	std_y = std(y)
	
	return covariance / (std_x * std_y) if std_x is not None and std_y is not None else None

def exp(x: float, precision: int = 50) -> float:
	"""
	Calcule l'exponentielle e^x en utilisant la série de Taylor.
	
	Args:
		x: Valeur à exponentier
		precision: Nombre de termes de la série (plus = plus précis)
	
	Returns:
		e^x
	"""

	try:
		if precision <= 0:
			raise ValueError("La précision doit être un entier positif")
		if x > 700:
			raise ValueError("La valeur est trop grande")
		if x < -700:
			raise ValueError("La valeur est trop petite")
		result = 1.0
		term = 1.0
		
		for n in range(1, precision):
			term *= x / n
			result += term
			
			if abs(term) < 1e-15:
				break
		return result
	except Exception as e:
		print(f"Erreur lors du calcul de l'exponentielle: {e}")
		return None

def ln(x: float, precision: int = 50) -> float:
	"""
	Calcule le logarithme naturel ln(x) en utilisant la série de Taylor.
	
	Args:
		x: Valeur à logarithmiser (doit être > 0)
		precision: Nombre de termes de la série (plus = plus précis)
	
	Returns:
		ln(x) ou None en cas d'erreur
	"""
	try:
		if x <= 0:
			raise ValueError("x doit être strictement positif pour ln(x)")
		if x == 1.0:
			return 0.0
		
		# Utiliser la série de Taylor pour ln(1+u) où u = x-1
		# ln(1+u) = u - u²/2 + u³/3 - u⁴/4 + ...
		# Cette série converge seulement pour |u| < 1, donc |x-1| < 1, donc 0 < x < 2
		
		# Si x >= 2, utiliser ln(x) = ln(2) + ln(x/2)
		if x >= 2.0:
			ln2 = 0.6931471805599453  # ln(2) précalculé
			# Diviser x par 2 jusqu'à ce qu'il soit < 2
			div_count = 0
			temp_x = x
			while temp_x >= 2.0:
				temp_x /= 2.0
				div_count += 1
			return div_count * ln2 + _ln_small(temp_x, precision)
		
		# Si x < 0.5, utiliser ln(x) = -ln(1/x)
		if x < 0.5:
			return -_ln_small(1.0 / x, precision)
		
		# Sinon, utiliser directement la série
		return _ln_small(x, precision)
		
	except Exception as e:
		print(f"Erreur lors du calcul de ln: {e}")
		return None

def _ln_small(x: float, precision: int = 50) -> float:
	"""Helper pour calculer ln(x) quand x est proche de 1 (0.5 < x < 2)"""
	u = x - 1.0
	result = 0.0
	sign = 1
	u_power = u
	
	for n in range(1, precision):
		term = sign * u_power / n
		result += term
		
		if abs(term) < 1e-15:
			break
		
		u_power *= u
		sign *= -1
	
	return result

def sigmoid(value: float) -> float:
	"""
	Calcule la sigmoïde d'une valeur.
	
	Args:
		value: Valeur à sigmoïde
	
	Returns:
		Sigmoïde de la valeur
	"""
	try:
		return 1 / (1 + exp(-value))
	except Exception as e:
		print(f"Erreur lors du calcul de la sigmoïde: {e}")
		return None

def calculate_homogeneity(scores_by_class: dict[str, dict[str, List[float]]]) -> dict[str, float]:
	"""
	Calcule l'homogénéité d'un ensemble de scores par colonne.
	
	Args:
		scores_by_class: Dictionnaire de scores par colonne
	
	Returns:
		Dictionnaire de l'homogénéité par colonne
	"""
	final_score = {}
	homogeneity = {}

	for column_name in scores_by_class:
		final_score[column_name] = {column_value: mean(scores_by_class[column_name][column_value]) for column_value in scores_by_class[column_name]}
	for column_value in list(final_score.values())[0].keys():
		means_per_column_value = [final_score[column_name][column_value] for column_name in final_score.keys()]
		homogeneity[column_value] = std(means_per_column_value)
	return homogeneity

def return_homogeneity_after_gap(homogeneity: dict[str, float]) -> dict[str, float]:
	"""
	Retourne les cours avec homogénéité au-dessus du plus grand gap.
	
	Args:
		homogeneity: Dictionnaire de l'homogénéité par valeur de colonne
	
	Returns:
		Dictionnaire des cours discriminants (après le gap)
	"""
	threshold = find_gap(homogeneity)
	homogeneity_after_gap = {}
	
	for column_name in homogeneity:
		if homogeneity[column_name] > threshold:
			homogeneity_after_gap[column_name] = homogeneity[column_name]
	
	print(f"\n📊 Gap détecté : seuil = {threshold:.4f}")
	print(f"✅ {len(homogeneity_after_gap)} cours sélectionnés (au-dessus du gap)")
	print(f"❌ {len(homogeneity) - len(homogeneity_after_gap)} cours exclus (en dessous du gap)")
	
	return homogeneity_after_gap

def find_gap(homogeneity: dict[str, float]) -> float:
	"""
	Trouve la valeur seuil correspondant au plus grand gap dans l'homogénéité.
	
	Args:
		homogeneity: Dictionnaire de l'homogénéité par valeur de colonne
	
	Returns:
		Valeur seuil (valeur AVANT le plus grand gap)
	"""
	homogeneity_values = list(homogeneity.values())
	homogeneity_values.sort()
	
	if len(homogeneity_values) < 2:
		return 0.0
	
	max_gap = 0.0
	gap_index = 0
	
	# Trouver le plus grand écart entre valeurs consécutives
	for i in range(1, len(homogeneity_values)):
		current_gap = homogeneity_values[i] - homogeneity_values[i - 1]
		if current_gap > max_gap:
			max_gap = current_gap
			gap_index = i - 1  # Index de la valeur AVANT le gap
	
	# Retourner la valeur AVANT le plus grand gap (seuil)
	threshold = homogeneity_values[gap_index]
	
	return threshold

def compute_correlation_matrix(column_names: List[str], column_data: List[List], 
							   exclude_columns: Optional[List[str]] = None) -> Optional[dict[str, dict[str, float]]]:
	"""
	Calcule la matrice de corrélation entre toutes les paires de colonnes.
	
	Args:
		column_names: Liste des noms de colonnes
		column_data: Liste de listes (une liste par colonne)
		exclude_columns: Liste des colonnes à exclure du calcul
	
	Returns:
		Dictionnaire {col1: {col2: corr, ...}, ...} ou None en cas d'erreur
	"""
	try:
		if exclude_columns is None:
			exclude_columns = []
		
		# Filtrer les colonnes
		filtered_indices = []
		filtered_names = []
		for i, name in enumerate(column_names):
			if name not in exclude_columns and i < len(column_data):
				filtered_indices.append(i)
				filtered_names.append(name)
		
		correlation_matrix = {}
		
		for i, name1 in enumerate(filtered_names):
			correlation_matrix[name1] = {}
			idx1 = filtered_indices[i]
			
			for j, name2 in enumerate(filtered_names):
				idx2 = filtered_indices[j]
				
				# Filtrer les données pour enlever les None
				clean_data1 = []
				clean_data2 = []
				for k in range(len(column_data[idx1])):
					if k < len(column_data[idx2]) and column_data[idx1][k] is not None and column_data[idx2][k] is not None:
						clean_data1.append(column_data[idx1][k])
						clean_data2.append(column_data[idx2][k])
				
				if len(clean_data1) > 1:
					corr = correlation_coefficient(clean_data1, clean_data2)
					correlation_matrix[name1][name2] = corr if corr is not None else 0.0
				else:
					correlation_matrix[name1][name2] = 0.0
		
		return correlation_matrix
		
	except Exception as e:
		print(f"Erreur lors du calcul de la matrice de corrélation: {e}")
		return None

def select_least_correlated_features(column_names: List[str], column_data: List[List],
									 max_correlation: float = 0.8,
									 exclude_columns: Optional[List[str]] = None) -> Optional[List[str]]:
	"""
	Sélectionne les features les moins corrélées entre elles.
	
	Stratégie: Si deux colonnes ont une corrélation absolue > max_correlation, on garde seulement
	la première dans l'ordre. Les corrélations négatives fortes sont aussi considérées comme
	problématiques (multicollinéarité) car elles indiquent une relation linéaire inversée.
	
	Args:
		column_names: Liste des noms de colonnes
		column_data: Liste de listes (une liste par colonne)
		max_correlation: Seuil de corrélation absolue maximum autorisé (0.0 à 1.0)
		                 Exemple: 0.7 signifie que si |corr| > 0.7, on exclut une des deux features
		exclude_columns: Liste des colonnes à exclure (ex: "Index", "Hogwarts House")
	
	Returns:
		Liste des noms de colonnes sélectionnées (moins corrélées)
	"""
	try:
		if exclude_columns is None:
			exclude_columns = []
		
		# Calculer la matrice de corrélation
		corr_matrix = compute_correlation_matrix(column_names, column_data, exclude_columns)
		if corr_matrix is None:
			return None
		
		# Filtrer les colonnes à considérer
		candidate_names = [name for name in column_names if name not in exclude_columns]
		
		selected_features = []
		removed_features = []
		
		for i, name1 in enumerate(candidate_names):
			if name1 not in corr_matrix:
				continue
			
			# Vérifier si cette feature est trop corrélée avec une feature déjà sélectionnée
			should_add = True
			for name2 in selected_features:
				if name2 in corr_matrix and name1 in corr_matrix[name2]:
					actual_corr = corr_matrix[name2][name1]
					abs_corr = abs(actual_corr)
					if abs_corr > max_correlation:
						should_add = False
						removed_features.append((name1, name2, actual_corr))  # Stocker la vraie corrélation
						break
			
			if should_add:
				selected_features.append(name1)
		
		return selected_features
		
	except Exception as e:
		print(f"Erreur lors de la sélection des features: {e}")
		return None

def print_correlation_matrix(column_names: List[str], column_data: List[List],
							  exclude_columns: Optional[List[str]] = None,
							  show_all: bool = False) -> bool:
	"""
	Affiche la matrice de corrélation de manière lisible.
	
	Args:
		column_names: Liste des noms de colonnes
		column_data: Liste de listes (une liste par colonne)
		exclude_columns: Liste des colonnes à exclure
		show_all: Si True, affiche toutes les corrélations, sinon seulement les fortes
	
	Returns:
		True si succès, False sinon
	"""
	try:
		corr_matrix = compute_correlation_matrix(column_names, column_data, exclude_columns)
		if corr_matrix is None:
			return False
		
		names = sorted(corr_matrix.keys())
		
		print("\n" + "="*80)
		print("MATRICE DE CORRÉLATION")
		print("="*80)
		
		if show_all:
			# Afficher toutes les corrélations
			print(f"\n{'':<20}", end="")
			for name in names:
				print(f"{name[:10]:>10}", end="")
			print()
			
			for name1 in names:
				print(f"{name1[:20]:<20}", end="")
				for name2 in names:
					corr = corr_matrix[name1][name2] if name2 in corr_matrix[name1] else 0.0
					print(f"{corr:>10.3f}", end="")
				print()
		else:
			# Afficher seulement les corrélations fortes (> 0.5 ou < -0.5)
			print("\n🔴 Corrélations fortes (|r| > 0.5):")
			print(f"{'Feature 1':<30} {'Feature 2':<30} {'Corrélation':<15}")
			print("-"*75)
			
			strong_corrs = []
			for i, name1 in enumerate(names):
				for j, name2 in enumerate(names):
					if i < j:  # Éviter les doublons
						corr = corr_matrix[name1][name2] if name2 in corr_matrix[name1] else 0.0
						if corr is not None and abs(corr) > 0.5:
							strong_corrs.append((name1, name2, corr))
			
			# Trier par valeur absolue de corrélation
			strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
			
			for name1, name2, corr in strong_corrs:
				indicator = "🔴" if abs(corr) > 0.8 else "🟡"
				print(f"{indicator} {name1:<28} {name2:<28} {corr:>10.4f}")
			
			# Afficher un résumé des corrélations maximales par feature
			print("\n📊 Résumé des corrélations maximales par feature (valeur absolue):")
			print("   Note: Une corrélation négative forte = corrélation forte inversée, pas une absence de corrélation")
			print(f"{'Feature':<30} {'|Corrélation| max':<20} {'Corrélation':<15} {'Avec':<30}")
			print("-"*95)
			
			feature_max_corr = []
			for name1 in names:
				max_abs_corr = 0.0
				max_corr = 0.0
				max_corr_with = None
				for name2 in names:
					if name1 != name2:
						corr = corr_matrix[name1][name2] if name2 in corr_matrix[name1] else 0.0
						if corr is not None and abs(corr) > abs(max_abs_corr):
							max_abs_corr = abs(corr)
							max_corr = corr
							max_corr_with = name2
				
				feature_max_corr.append((name1, max_abs_corr, max_corr, max_corr_with))
			
			# Trier par valeur absolue de corrélation maximale (croissant = moins corrélé)
			feature_max_corr.sort(key=lambda x: x[1])  # x[1] est la valeur absolue
			
			for name, max_abs_corr, max_corr, with_name in feature_max_corr:
				if max_abs_corr < 0.3:
					indicator = "✅"  # Peu corrélée = bonne feature (indépendante)
				elif max_abs_corr < 0.5:
					indicator = "🟢"  # Corrélation modérée
				elif max_abs_corr < 0.7:
					indicator = "🟡"  # Corrélation forte
				else:
					indicator = "🔴"  # Très forte corrélation (multicollinéarité)
				
				# Afficher le signe de la corrélation pour comprendre la relation
				sign = "+" if max_corr >= 0 else "-"
				
				if with_name:
					print(f"{indicator} {name:<28} {max_abs_corr:>10.4f}      {sign}{max_abs_corr:>10.4f}      {with_name:<30}")
				else:
					print(f"{indicator} {name:<28} {max_abs_corr:>10.4f}      {sign}{max_abs_corr:>10.4f}      {'N/A':<30}")
		
		print("="*80 + "\n")
		return True
		
	except Exception as e:
		print(f"Erreur lors de l'affichage de la matrice: {e}")
		return False

def calculate_cost_function(theta: List[float], selected_features_data: dict[str, dict[str, List[float]]], features_names: List[str]) -> float:
	"""
	Calcule la fonction de coût (cost function) pour la régression logistique.
	
	Args:
		theta: Liste des paramètres theta
		selected_features_data: Dictionnaire des données filtrées
		features_names: Liste des noms des features sélectionnées
	"""
	
	return 0.0