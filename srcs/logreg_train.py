import utils
import sys
from typing import Union, List, Optional, Tuple
import matplotlib.pyplot as plt

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
		clean_numeric_data = utils.none_filter(numeric_data)
		print(f"clean_numeric_data : {clean_numeric_data}")
		homogeneity = utils.calculate_homogeneity(clean_numeric_data)
		homogeneity_after_gap = utils.return_homogeneity_after_gap(homogeneity)
		print(f"homogeneity_after_gap : {homogeneity_after_gap}")
	except (Exception) as e:
		print(f"Error: {e}")
		return False
	return True

if __name__ == "__main__":
	logreg_train()