import utils
import sys
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt

def histogram():
	try:
		if (len(sys.argv) != 2):
			raise AttributeError("You must input 1 path")
		header, df = utils.load_csv(sys.argv[1], columns=None, skip_header=True, 
		                      return_header=True, auto_convert=True)
		if df is None:
			raise Exception("Failed to load data")
		courses_names, courses_data = utils.extract_numeric_columns(sys.argv[1], df, skip_header=True)
		courses_names.remove("Index")
		if courses_names is None or courses_data is None:
			raise Exception("Failed to extract courses data")
		houses_scores = create_houses_tab_scores(header, df, courses_names)
		courses_homogeneity = calculat_courses_homogeneity(houses_scores)
		plot_courses_homogeneity(courses_homogeneity)		

	except (Exception) as e:
		print(f"Error: {e}")

def create_houses_tab_scores(header: List[str], data: List[List], courses_names: List[str]) -> dict[str, dict[str, List[float]]]:
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
	# print(f"houses_scores : {houses_scores}")
	return houses_scores

def calculat_courses_homogeneity(houses_scores: dict[str, dict[str, List[float]]]) -> dict[str, float]:
	final_score = {}
	courses_homogeneity = {}

	for house in houses_scores:
		final_score[house] = {course: utils.mean(houses_scores[house][course]) for course in houses_scores[house]}
	for course in list(final_score.values())[0].keys():
		means_per_house = [final_score[house][course] for house in final_score.keys()]
		courses_homogeneity[course] = utils.std(means_per_house)
	print(f"courses_homogeneity : {courses_homogeneity}")
	return courses_homogeneity

def plot_courses_homogeneity(courses_homogeneity: dict[str, float]):
	"""Affiche l'écart-type des moyennes pour tous les cours"""
	
	# Trier les cours par homogénéité (plus petit = plus homogène)
	sorted_courses = sorted(courses_homogeneity.items(), key=lambda x: x[1])
	course_names = [item[0] for item in sorted_courses]
	homogeneity_scores = [item[1] for item in sorted_courses]
	
	plt.figure(figsize=(14, 8))
	
	# Créer le graphique en barres
	bars = plt.bar(range(len(course_names)), homogeneity_scores, 
	               color=plt.cm.viridis([score/max(homogeneity_scores) for score in homogeneity_scores]))
	
	# Personnaliser l'affichage
	plt.title("Homogénéité des cours de Poudlard\n(Écart-type des moyennes inter-maisons)", 
	          fontsize=14, fontweight='bold')
	plt.xlabel("Cours", fontsize=12)
	plt.ylabel("Écart-type des moyennes", fontsize=12)
	
	# Rotation des labels pour la lisibilité
	plt.xticks(range(len(course_names)), course_names, rotation=45, ha='right')
	
	# Ajouter les valeurs sur les barres
	for i, (bar, score) in enumerate(zip(bars, homogeneity_scores)):
		plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
		         f'{score:.4f}', ha='center', va='bottom', fontsize=9)
	
	# Mettre en évidence le cours le plus homogène
	most_homogeneous_idx = 0  # Premier élément après tri
	bars[most_homogeneous_idx].set_color('red')
	bars[most_homogeneous_idx].set_alpha(0.8)
	
	# Ajouter une flèche qui pointe vers la barre la plus homogène
	most_homogeneous_score = homogeneity_scores[most_homogeneous_idx]
	most_homogeneous_course = course_names[most_homogeneous_idx]
	
	# Position de la flèche (au-dessus de la barre)
	arrow_x = most_homogeneous_idx
	arrow_y = most_homogeneous_score + max(homogeneity_scores) * 0.1  # 10% au-dessus de la barre
	
	# Position du texte
	text_x = arrow_x
	text_y = arrow_y + max(homogeneity_scores) * 0.05
	
	# Dessiner la flèche
	plt.annotate(f'Plus homogène\n{most_homogeneous_course}\n({most_homogeneous_score:.4f})',
	             xy=(arrow_x, most_homogeneous_score),
	             xytext=(text_x, text_y),
	             arrowprops=dict(arrowstyle='->', color='red', lw=2),
	             fontsize=10, fontweight='bold', color='red',
	             ha='center', va='bottom',
	             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
	
	plt.grid(True, alpha=0.3, axis='y')
	plt.tight_layout()
	
	# Sauvegarder
	plt.savefig("courses_homogeneity.png", dpi=300, bbox_inches='tight')
	plt.show()
	
	# Afficher le classement
	print("\n=== Classement des cours par homogénéité (plus petit = plus homogène) ===")
	for i, (course, score) in enumerate(sorted_courses, 1):
		status = "🏆 PLUS HOMOGÈNE" if i == 1 else ""
		print(f"{i:2d}. {course:<25} : {score:.6f} {status}")
	
	return True

if __name__ == "__main__":
	histogram()