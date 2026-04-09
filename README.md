# 🎓 DSLR - Classification des Maisons de Poudlard - Cours Complet

**Projet de machine learning from scratch** pour classifier les maisons de Poudlard (Gryffindor, Hufflepuff, Ravenclaw, Slytherin) à partir des notes des étudiants en utilisant la **régression logistique**.

---

## 📋 Table des matières

1. [Introduction](#-introduction)
2. [Concepts fondamentaux](#-concepts-fondamentaux)
3. [Mathématiques de la régression logistique](#-mathématiques-de-la-régression-logistique)
4. [Guide d'implémentation pas à pas](#-guide-dimplémentation-pas-à-pas)
5. [Quand utiliser chaque formule](#-quand-utiliser-chaque-formule)
6. [Convergence et Patience](#-convergence-et-patience)
7. [Structure du projet](#-structure-du-projet)
8. [Utilisation](#-utilisation)

---

## 🎯 Introduction

### Objectif du projet

Implémenter une régression logistique **from scratch** (sans bibliothèques ML) pour prédire la maison d'un étudiant de Poudlard à partir de ses notes dans différents cours.

### Classification binaire vs multiclasse

#### ⚠️ IMPORTANT : La régression logistique est TOUJOURS binaire

**La régression logistique de base ne peut prédire que 2 classes : 0 ou 1.**

Toutes les formules du README (fonction de coût, gradient, etc.) supposent que **y ∈ {0, 1}**.

#### Pour 4 maisons : One-vs-All (4 modèles binaires)

Pour classifier 4 maisons, on entraîne **4 modèles binaires séparés** :

- **Modèle 0 (Gryffindor)** : "Est-ce Gryffindor ?" → y_binary = 1 si y == 0, sinon y_binary = 0
- **Modèle 1 (Hufflepuff)** : "Est-ce Hufflepuff ?" → y_binary = 1 si y == 1, sinon y_binary = 0
- **Modèle 2 (Ravenclaw)** : "Est-ce Ravenclaw ?" → y_binary = 1 si y == 2, sinon y_binary = 0
- **Modèle 3 (Slytherin)** : "Est-ce Slytherin ?" → y_binary = 1 si y == 3, sinon y_binary = 0

**Chaque modèle utilise y_binary (0 ou 1), pas y multiclasse (0, 1, 2, 3) !**

#### Exemple concret

```python
# Données originales (multiclasse)
y = [0, 0, 1, 2, 3, 0]  # 0=Gryffindor, 1=Hufflepuff, 2=Ravenclaw, 3=Slytherin

# Pour le modèle Gryffindor (house_idx = 0)
y_binary = [1, 1, 0, 0, 0, 1]  # 1 si Gryffindor, 0 sinon

# Pour le modèle Hufflepuff (house_idx = 1)
y_binary = [0, 0, 1, 0, 0, 0]  # 1 si Hufflepuff, 0 sinon
```

**Toutes les formules du README utilisent y_binary (0 ou 1), pas y multiclasse !**

---

### 🔍 Utilisation de y_binary dans le calcul du coût - EXPLICATION DÉTAILLÉE

#### Le problème

La formule de la fonction de coût **ne fonctionne QUE avec y ∈ {0, 1}** :

```python
J(θ) = -1/m × Σ[yᵢ × log(h) + (1-yᵢ) × log(1-h)]
```

Mais vous avez **4 maisons** : `y = [0, 1, 2, 3, 0, 1, ...]`

#### La solution : Conversion y → y_binary pour chaque modèle

**Pour CHAQUE modèle**, on convertit y multiclasse en y_binary :

```python
# Données originales
X = [[10, 8], [12, 9], [5, 3], [7, 6], [14, 11]]  # 5 étudiants
y = [0, 0, 1, 2, 3]  # 0=Gryffindor, 1=Hufflepuff, 2=Ravenclaw, 3=Slytherin
m = 5

# ========== MODÈLE GRYFFINDOR (house_idx = 0) ==========
# Question : "Est-ce Gryffindor ?" → OUI (1) ou NON (0)
y_binary_gryff = [1, 1, 0, 0, 0]
#                 ↑  ↑  ↑  ↑  ↑
#                 G  G  H  R  S
#               OUI OUI NON NON NON

# ========== MODÈLE HUFFLEPUFF (house_idx = 1) ==========
# Question : "Est-ce Hufflepuff ?" → OUI (1) ou NON (0)
y_binary_huff = [0, 0, 1, 0, 0]
#                ↑  ↑  ↑  ↑  ↑
#                G  G  H  R  S
#              NON NON OUI NON NON

# ========== MODÈLE RAVENCLAW (house_idx = 2) ==========
y_binary_raven = [0, 0, 0, 1, 0]

# ========== MODÈLE SLYTHERIN (house_idx = 3) ==========
y_binary_slyth = [0, 0, 0, 0, 1]
```

#### Où et comment y_binary est utilisé dans le calcul du coût

```python
def logistic_regression_binary(X, y_binary, m, n_features):
	"""
	Entraîne UN SEUL modèle binaire (appelé 4 fois pour 4 maisons).
	
	Args:
		X: Features (IDENTIQUE pour tous les modèles)
		y_binary: Labels binaires (DIFFÉRENT pour chaque modèle)
		m: Nombre d'étudiants
		n_features: Nombre de features
	"""
	
	theta = [0.0] * (n_features + 1)
	learning_rate = 0.01
	
	for epoch in range(max_epochs):
		
		# 1. CALCULER LES PRÉDICTIONS
		predictions = []
		for i in range(m):
			z = theta[0] * 1  # Biais
			for j in range(n_features):
				z += theta[j+1] * X[i][j]
			h = sigmoid(z)
			predictions.append(h)
		
		# 2. CALCULER LE COÛT ← ICI ON UTILISE y_binary !
		cost = 0.0
		for i in range(m):
			h = predictions[i]
			
			# ⚠️ IMPORTANT : On utilise y_binary[i], PAS y[i] !
			if y_binary[i] == 1:  # Si c'est la bonne maison (OUI)
				cost += ln(h)
			else:  # Si ce n'est pas la bonne maison (NON)
				cost += ln(1.0 - h)
		
		cost = -(1.0 / m) * cost
		
		# 3. CALCULER LE GRADIENT ← ICI AUSSI !
		gradient = [0.0] * len(theta)
		for j in range(len(theta)):
			sum_error = 0.0
			for i in range(m):
				# ⚠️ IMPORTANT : On utilise y_binary[i], PAS y[i] !
				error = predictions[i] - y_binary[i]
				x_ij = 1.0 if j == 0 else X[i][j-1]
				sum_error += error * x_ij
			gradient[j] = (1.0 / m) * sum_error
		
		# 4. METTRE À JOUR THETA
		for j in range(len(theta)):
			theta[j] = theta[j] - learning_rate * gradient[j]
	
	return theta
```

#### Exemple concret étape par étape pour le modèle Gryffindor

```python
# Données
X = [[10, 8], [12, 9], [5, 3], [7, 6], [14, 11]]
y = [0, 0, 1, 2, 3]  # Multiclasse original
m = 5

# CONVERSION pour le modèle Gryffindor
y_binary = [1, 1, 0, 0, 0]
#           ↑  ↑  ↑  ↑  ↑
#         Gryff Gryff Huff Raven Slyth

# EPOCH 0 : theta = [0, 0, 0]
predictions = [0.5, 0.5, 0.5, 0.5, 0.5]  # sigmoid(0) = 0.5

# CALCUL DU COÛT
cost = 0.0

# Étudiant 0 : y_binary[0] = 1 (Gryffindor), h = 0.5
cost += ln(0.5)  # ≈ -0.693

# Étudiant 1 : y_binary[1] = 1 (Gryffindor), h = 0.5
cost += ln(0.5)  # ≈ -0.693

# Étudiant 2 : y_binary[2] = 0 (Hufflepuff, pas Gryffindor), h = 0.5
cost += ln(1.0 - 0.5)  # ln(0.5) ≈ -0.693

# Étudiant 3 : y_binary[3] = 0 (Ravenclaw, pas Gryffindor), h = 0.5
cost += ln(1.0 - 0.5)  # ln(0.5) ≈ -0.693

# Étudiant 4 : y_binary[4] = 0 (Slytherin, pas Gryffindor), h = 0.5
cost += ln(1.0 - 0.5)  # ln(0.5) ≈ -0.693

# Total
cost = -(1.0 / 5) * (-0.693 - 0.693 - 0.693 - 0.693 - 0.693)
cost = -(1.0 / 5) * (-3.465)
cost ≈ 0.693

# CALCUL DU GRADIENT
# Pour θ₀ (biais)
gradient[0] = (1/5) × [(0.5-1)×1 + (0.5-1)×1 + (0.5-0)×1 + (0.5-0)×1 + (0.5-0)×1]
            = (1/5) × [-0.5 - 0.5 + 0.5 + 0.5 + 0.5]
            = 0.1

# MISE À JOUR
theta[0] = 0 - 0.01 × 0.1 = -0.001
# ... et ainsi de suite pour θ₁, θ₂
```

#### Détails de la formule

Dans la formule du coût, selon y_binary :

```python
# Forme complète
cost += y_binary[i] × ln(h) + (1 - y_binary[i]) × ln(1 - h)

# Si y_binary[i] = 1 (c'est la bonne maison) :
cost += 1 × ln(h) + 0 × ln(1-h)
cost += ln(h)  # Pénalise si h est faible (mauvaise prédiction)

# Si y_binary[i] = 0 (ce n'est PAS la bonne maison) :
cost += 0 × ln(h) + 1 × ln(1-h)
cost += ln(1-h)  # Pénalise si h est élevé (mauvaise prédiction)
```

#### Entraîner les 4 modèles

```python
def train_one_vs_all(X, y, m, n_features):
	"""Entraîne 4 modèles binaires (One-vs-All)"""
	
	all_thetas = []
	
	for house_idx in range(4):  # 0, 1, 2, 3
		# CONVERSION : y multiclasse → y_binary
		y_binary = [1 if y[i] == house_idx else 0 for i in range(m)]
		
		# Entraîner CE modèle avec y_binary
		theta = logistic_regression_binary(X, y_binary, m, n_features)
		all_thetas.append(theta)
	
	return all_thetas  # 4 vecteurs theta (un par maison)
```

#### Points clés à retenir

1. **Chaque modèle a son propre y_binary** :
   - Modèle Gryffindor : `[1, 1, 0, 0, 0]`
   - Modèle Hufflepuff : `[0, 0, 1, 0, 0]`
   - etc.

2. **Tous les modèles utilisent le même X** (features identiques)

3. **y_binary est TOUJOURS 0 ou 1** (jamais 2 ou 3)

4. **Chaque modèle répond à SA question binaire** :
   - "Est-ce Gryffindor ?" → y_binary spécifique
   - "Est-ce Hufflepuff ?" → y_binary différent
   - etc.

5. **Dans la fonction de coût et le gradient, on utilise TOUJOURS y_binary, jamais y multiclasse**

### Différence avec la régression linéaire

- **Régression linéaire** : Sortie continue (ex: prix = 1000 + 50×taille)
- **Régression logistique** : Sortie = probabilité entre 0 et 1

---

## 📚 Concepts fondamentaux

### 1. Le biais (bias) - θ₀

#### Définition

Le **biais (θ₀)** est un terme constant dans le modèle qui permet de décaler la fonction de décision.

**Analogie :** Imaginez une balance :
- Les features (x₁, x₂, ...) sont les poids qu'on met sur la balance
- Le **biais** est comme un poids de base qu'on met toujours sur un côté
- Même si toutes les features sont à zéro, le biais peut faire pencher la balance

#### Pourquoi on en a besoin ?

**Sans biais :**
```python
h = sigmoid(θ₁×x₁ + θ₂×x₂)
# Si toutes les notes sont à 0
h = sigmoid(0.3×0 + 0.2×0) = sigmoid(0) = 0.5
→ Toujours 50% de probabilité quand tout est à zéro ❌
```

**Avec biais :**
```python
h = sigmoid(θ₀×1 + θ₁×x₁ + θ₂×x₂)
# Si toutes les notes sont à 0
h = sigmoid(0.5×1 + 0.3×0 + 0.2×0) = sigmoid(0.5) ≈ 0.62
→ 62% de probabilité même avec des notes nulles ✅
→ Le biais "dit" : "Même sans notes, il y a une tendance à être Gryffindor"
```

#### En résumé

- **Biais = terme constant** qui ne dépend pas des features
- **Permet de décaler** la fonction de décision
- **Indispensable** pour avoir un modèle flexible
- **Toujours multiplié par 1** (d'où x₀ = 1)

---

### 2. La sigmoïde - g(z)

#### Définition

La sigmoïde transforme n'importe quel nombre en une valeur entre 0 et 1.

**Formule :**
```
g(z) = 1 / (1 + e^(-z))
```

Où `e` est le nombre d'Euler (≈ 2.718).

#### Propriétés

- Résultat toujours entre 0 et 1
- Si z → -∞ → g(z) → 0
- Si z = 0 → g(z) = 0.5
- Si z → +∞ → g(z) → 1
- Forme en "S" (courbe sigmoïde)
- Dérivable partout (important pour le gradient)
- Fonction monotone (croissante)

#### Pourquoi l'utiliser ?

- Permet d'interpréter le résultat comme une probabilité
- Dérivable partout (important pour le gradient)
- Fonction monotone (croissante)

#### Forme de la courbe

```
g(z)
 1 |     ╱────────────
   |    ╱
0.5|───╱
   | ╱
 0 |╱
   └───────────→ z
```

---

### 3. Fonction de prédiction - h_θ(x)

#### Définition

La fonction de prédiction calcule la probabilité qu'un exemple appartienne à la classe 1.

**Formule :**
```
h_θ(x) = g(θᵀ · x) = sigmoid(θ₀×1 + θ₁×x₁ + θ₂×x₂ + ...)
```

#### Interprétation

- `h_θ(x) = 0.9` → 90% de chance d'être dans la classe 1
- `h_θ(x) = 0.1` → 10% de chance (donc 90% d'être dans la classe 0)
- `h_θ(x) = 0.5` → Incertitude maximale

#### Exemple concret

```python
theta = [0.5, 0.3, -0.2]  # [θ₀ (bias), θ₁, θ₂]
x = [1, 10, 8]            # [x₀=1 (ajouté), x₁=10, x₂=8]

z = 0.5×1 + 0.3×10 + (-0.2)×8 = 1.9
h = sigmoid(1.9) ≈ 0.87

→ 87% de probabilité d'être dans la classe 1
```

#### ⚠️ IMPORTANT : x₀ = 1

**Les données brutes n'ont pas x₀**, on doit l'ajouter artificiellement :

```python
# Données brutes
features_brutes = [10, 8]  # Notes d'Arithmancy et Astronomy

# ⚠️ Ajouter x₀ = 1 au début pour le biais
x = [1] + features_brutes  # [x₀=1 (ajouté), x₁=10, x₂=8]
# Résultat : x = [1, 10, 8]
```

**Pourquoi x₀ = 1 ?**
- x₀ est toujours égal à 1, il sert uniquement à multiplier θ₀ (le biais)
- Cela permet d'avoir un terme constant dans notre modèle : `θ₀×1 + θ₁×x₁ + θ₂×x₂`
- Sans ce 1, on ne pourrait pas avoir de biais indépendant des features

---

### 4. Fonction de coût - J(θ)

#### Définition

La fonction de coût mesure à quel point le modèle fait des erreurs. Plus elle est faible, meilleur est le modèle.

**Formule :**
```
J(θ) = -1/m × Σ[i=1 à m] [yᵢ × log(h_θ(xᵢ)) + (1-yᵢ) × log(1-h_θ(xᵢ))]
```

Où :
- **J(θ)** = fonction de coût (cost function)
- **m** = nombre d'exemples d'entraînement
- **yᵢ** = label réel **BINAIRE** (0 ou 1) pour l'exemple i
  - ⚠️ **IMPORTANT** : Cette formule fonctionne UNIQUEMENT avec y ∈ {0, 1}
  - Pour One-vs-All, on utilise y_binary (0 ou 1), pas y multiclasse (0, 1, 2, 3)
- **h_θ(xᵢ)** = probabilité prédite pour l'exemple i
- **log** = logarithme naturel (ln, base e)

#### Interprétation

- **Coût faible** (proche de 0) → Modèle fait peu d'erreurs ✅
- **Coût élevé** (grand) → Modèle fait beaucoup d'erreurs ❌
- Objectif : minimiser J(θ)

#### Pourquoi cette formule ?

C'est la **log-loss** (entropie croisée). Elle pénalise fortement les mauvaises prédictions.

**Cas 1 : y = 1 (vraie classe = 1)**
```
Si h_θ(x) = 0.9 (bonne prédiction) :
  Terme = 1 × log(0.9) ≈ -0.105  → Coût faible ✅

Si h_θ(x) = 0.1 (mauvaise prédiction) :
  Terme = 1 × log(0.1) ≈ -2.303  → Coût élevé ❌
```

**Cas 2 : y = 0 (vraie classe = 0)**
```
Si h_θ(x) = 0.1 (bonne prédiction) :
  Terme = 0 × log(0.1) + 1 × log(0.9) ≈ -0.105  → Coût faible ✅

Si h_θ(x) = 0.9 (mauvaise prédiction) :
  Terme = 0 × log(0.9) + 1 × log(0.1) ≈ -2.303  → Coût élevé ❌
```

---

### 5. Le gradient - ∂J/∂θⱼ

#### Définition

Le gradient est la dérivée partielle de la fonction de coût par rapport à un paramètre. Il indique dans quelle direction et de combien ajuster le paramètre pour réduire le coût.

**Formule :**
```
∂J/∂θⱼ = 1/m × Σ[i=1 à m] (h_θ(xᵢ) - yᵢ) × xᵢⱼ
```

Où :
- **j** = indice du paramètre (0, 1, 2, ..., n)
  - j = 0 → biais (θ₀)
  - j = 1, 2, ..., n → poids des features
- **xᵢⱼ** = valeur de la feature j pour l'exemple i

#### Interprétation

- **Gradient positif** → Le paramètre est trop grand → Diminuer θⱼ
- **Gradient négatif** → Le paramètre est trop petit → Augmenter θⱼ
- **Gradient proche de 0** → Paramètre optimal (minimum)

#### Détails

- `(h_θ(xᵢ) - yᵢ)` = erreur de prédiction
  - Si h > y → erreur positive → on doit diminuer θⱼ
  - Si h < y → erreur négative → on doit augmenter θⱼ
- `xᵢⱼ` = poids de la feature
  - Si x est grand → ajustement plus important
  - Si x est petit → ajustement plus faible

#### Exemple

```python
# Pour θ₁ (poids de la feature 1)
# Exemple 1: h=0.8, y=1, x₁=10
Erreur = 0.8 - 1 = -0.2
Contribution = -0.2 × 10 = -2

# Exemple 2: h=0.3, y=0, x₁=5
Erreur = 0.3 - 0 = 0.3
Contribution = 0.3 × 5 = 1.5

# Gradient pour θ₁
∂J/∂θ₁ = (1/2) × (-2 + 1.5) = -0.25

→ On doit augmenter θ₁ (car gradient négatif)
```

---

### 6. Descente de gradient

#### Définition

Algorithme d'optimisation qui trouve les valeurs optimales des paramètres en les ajustant progressivement dans la direction opposée au gradient.

#### Principe

1. Commencer avec des paramètres initiaux (souvent à zéro)
2. Calculer le gradient pour chaque paramètre
3. Ajuster chaque paramètre : `θⱼ = θⱼ - α × gradient`
4. Répéter jusqu'à convergence

#### Formule de mise à jour

```
θⱼ = θⱼ - α × ∂J/∂θⱼ
```

**Pourquoi "-" ?**
- Le gradient pointe vers le maximum
- On veut aller vers le minimum
- Donc on va dans la direction opposée

**Analogie :**
Comme descendre une colline les yeux bandés :
- Le gradient = la pente (direction la plus raide)
- On va dans la direction opposée = on descend
- α = la taille de nos pas

---

### 7. Learning Rate - α (alpha)

#### Définition

Le learning rate contrôle la taille des pas lors de la mise à jour des paramètres.

#### Rôle

- **α trop petit** (ex: 0.0001) → Convergence très lente, beaucoup d'itérations
- **α optimal** (ex: 0.01) → Convergence rapide et stable
- **α trop grand** (ex: 1.0) → Risque de divergence, le modèle ne converge pas

#### Exemple

```python
# Avec α = 0.01
θ₁ = θ₁ - 0.01 × gradient  # Petit pas

# Avec α = 0.1
θ₁ = θ₁ - 0.1 × gradient   # Grand pas (attention à la convergence!)
```

#### Comment choisir ?

- Commencer par 0.01
- Si trop lent → augmenter légèrement
- Si le coût augmente → diminuer

---

### 8. Autres définitions importantes

#### Logarithme naturel (log ou ln)

**Propriétés importantes :**
- `log(1) = 0`
- `log(e) = 1` où e ≈ 2.718
- `log(a × b) = log(a) + log(b)`
- `log(a/b) = log(a) - log(b)`
- Si a < 1 → log(a) < 0 (négatif)
- Si a > 1 → log(a) > 0 (positif)

**Pourquoi dans la fonction de coût ?**
- Permet de transformer les produits en sommes
- Facilite la dérivation
- Pénalise exponentiellement les mauvaises prédictions

#### Entropie croisée (Cross-Entropy)

**Définition :**
L'entropie croisée est la fonction de coût utilisée en classification. Elle mesure la différence entre la distribution prédite et la distribution réelle.

**Pourquoi "entropie" ?**
- L'entropie mesure l'incertitude
- Plus la prédiction est certaine (proche de 0 ou 1), plus l'entropie est faible
- Plus la prédiction est incertaine (proche de 0.5), plus l'entropie est élevée

**Relation avec la log-loss :**
- La log-loss est l'entropie croisée pour la classification binaire
- C'est la même chose, juste un nom différent

#### Multicollinéarité

**Définition :**
La multicollinéarité se produit quand deux features ou plus sont très corrélées entre elles.

**Problème :**
- Les features redondantes n'apportent pas d'information supplémentaire
- Peut rendre le modèle instable
- Difficulté à interpréter les paramètres

**Solution :**
- Sélectionner les features les moins corrélées
- Supprimer les features avec |corrélation| > 0.7-0.8
- Utiliser la régularisation

#### Surapprentissage (Overfitting)

**Définition :**
Le surapprentissage se produit quand le modèle mémorise les données d'entraînement au lieu d'apprendre les patterns généraux.

**Symptômes :**
- Très bon sur les données d'entraînement
- Mauvais sur de nouvelles données
- Paramètres très grands

**Solutions :**
- Régularisation
- Plus de données d'entraînement
- Moins de features
- Arrêter l'entraînement plus tôt (early stopping)

#### Sous-apprentissage (Underfitting)

**Définition :**
Le sous-apprentissage se produit quand le modèle est trop simple et ne capture pas les patterns des données.

**Symptômes :**
- Mauvais sur les données d'entraînement
- Mauvais sur de nouvelles données
- Modèle trop simple

**Solutions :**
- Plus de features
- Modèle plus complexe
- Plus d'itérations d'entraînement
- Learning rate plus adapté

---

## 🔢 Mathématiques de la régression logistique

### Notation et variables

- **θ (theta)** : Vecteur des paramètres du modèle (poids/bias)
- **θ₀ (theta_0)** : Biais (bias) ou intercept
- **θⱼ (theta_j)** : Poids du paramètre j (j = 1, 2, ..., n)
- **x** : Vecteur des features (caractéristiques) d'un exemple
- **x₀ (x_0)** : Toujours égal à 1 (pour le biais)
- **xᵢ (x_i)** : Feature i (i = 1, 2, ..., n)
- **xᵢⱼ (x_i_j)** : Valeur de la feature j pour l'exemple i
- **y** : Label réel (valeur cible) - 0 ou 1 pour classification binaire
- **yᵢ (y_i)** : Label réel pour l'exemple i
- **h_θ(x)** : Fonction de prédiction (hypothèse) - probabilité prédite
- **g(z)** : Fonction sigmoïde
- **z** : Combinaison linéaire = θᵀ · x
- **J(θ)** : Fonction de coût (cost function)
- **m** : Nombre d'exemples dans le dataset d'entraînement
  - Exemple : Si vous avez 1000 étudiants dans votre dataset, m = 1000
  - Dans les boucles : `for i in range(m)` signifie "pour chaque exemple"
  - ⚠️ **IMPORTANT** : m est le nombre TOTAL d'étudiants, pas le nombre par cours ou par maison !
  - Chaque étudiant a plusieurs features (cours) mais compte comme 1 seul exemple
- **n** : Nombre de features (caractéristiques)
- **α (alpha)** : Learning rate (taux d'apprentissage)
- **λ (lambda)** : Paramètre de régularisation (optionnel)
- **e** : Nombre d'Euler ≈ 2.71828
- **log** : Logarithme naturel (ln, base e)
- **∂** : Symbole de dérivée partielle

---

### Résumé des formules clés

```
1. Sigmoïde :          g(z) = 1/(1+e^(-z))
   où z = θᵀ · x (produit scalaire)

2. Prédiction :        h_θ(x) = g(θᵀ · x)
   où h_θ(x) = probabilité prédite entre 0 et 1

3. Fonction de coût :  J(θ) = -1/m × Σ[y×log(h) + (1-y)×log(1-h)]
   où J(θ) = coût, m = nombre d'exemples, y = label réel, h = probabilité prédite

4. Gradient :          ∂J/∂θⱼ = 1/m × Σ(h - y) × xⱼ
   où ∂J/∂θⱼ = dérivée partielle, h = probabilité prédite, y = label réel, xⱼ = feature j

5. Mise à jour :       θⱼ = θⱼ - α × ∂J/∂θⱼ
   où θⱼ = paramètre, α = learning rate, ∂J/∂θⱼ = gradient
```

---

### Dérivation mathématique du gradient

#### Étape 1 : Décomposer la fonction de coût

```
J(θ) = -1/m × Σ [y × log(h) + (1-y) × log(1-h)]
```

Où `h = h_θ(x) = g(z)` et `z = θᵀ · x`.

#### Étape 2 : Dériver par rapport à θⱼ

On utilise la **règle de la chaîne** :
```
∂J/∂θⱼ = ∂J/∂h × ∂h/∂z × ∂z/∂θⱼ
```

#### Étape 3 : Calculer chaque terme

**a) ∂J/∂h :**
```
∂J/∂h = -1/m × [y/h - (1-y)/(1-h)]
     = -1/m × [(y - h) / (h(1-h))]
```

**b) ∂h/∂z :**
```
h = g(z) = 1/(1+e^(-z))

∂h/∂z = e^(-z) / (1+e^(-z))²
      = h × (1-h)
```

**c) ∂z/∂θⱼ :**
```
z = θ₀×1 + θ₁×x₁ + ... + θⱼ×xⱼ + ...

∂z/∂θⱼ = xⱼ
```

#### Étape 4 : Combiner

```
∂J/∂θⱼ = -1/m × [(y-h)/(h(1-h))] × [h(1-h)] × xⱼ
       = -1/m × (y-h) × xⱼ
       = 1/m × (h-y) × xⱼ
```

**Résultat final :**
```
∂J/∂θⱼ = 1/m × Σ (h_θ(xᵢ) - yᵢ) × xᵢⱼ
```

---

## 🛠️ Guide d'implémentation pas à pas

### Algorithme complet

```python
1. Préparer les données :
   - Pour chaque exemple, ajouter x₀ = 1 au début
   - Exemple : [10, 8] → [1, 10, 8]

2. Initialiser θ = [0, 0, 0, ..., 0]
   - Un paramètre pour le biais (θ₀) + un par feature (θ₁, θ₂, ...)

3. Pour chaque itération :
   a) Calculer h_θ(x) pour tous les exemples
      - z = θ₀×1 + θ₁×x₁ + θ₂×x₂ + ...
      - h = sigmoid(z)
   b) Calculer le gradient ∂J/∂θⱼ pour chaque paramètre
   c) Mettre à jour : θⱼ = θⱼ - α × ∂J/∂θⱼ
   d) (Optionnel) Calculer J(θ) pour suivre la progression

4. Répéter jusqu'à convergence
```

---

### Exemple complet simplifié

**Données brutes :**
```python
X_raw = [[10, 8], [12, 9], [5, 3]]  # 3 exemples, 2 features (sans le biais)
y = [1, 1, 0]                        # Labels
```

**⚠️ IMPORTANT : Ajouter x₀ = 1 pour le biais**

```python
# Données transformées (avec x₀=1 ajouté)
X = [
    [1, 10, 8],   # Exemple 1 : [x₀=1 (ajouté), x₁=10, x₂=8]
    [1, 12, 9],   # Exemple 2 : [x₀=1 (ajouté), x₁=12, x₂=9]
    [1, 5, 3]     # Exemple 3 : [x₀=1 (ajouté), x₁=5, x₂=3]
]
```

**Étape 1 : Initialiser**
```python
# theta (θ) = paramètres du modèle
# On a 3 paramètres : θ₀ (bias) + θ₁ (feature 1) + θ₂ (feature 2)
theta = [0, 0, 0]  # [θ₀ (bias), θ₁ (poids feature 1), θ₂ (poids feature 2)]

# alpha (α) = learning rate
alpha = 0.01
```

**Étape 2 : Calculer les prédictions**
```python
# Exemple 1: x = [x₀=1, x₁=10, x₂=8]
z_1 = theta[0]×1 + theta[1]×10 + theta[2]×8
    = 0×1 + 0×10 + 0×8
    = 0
h_1 = sigmoid(z_1) = sigmoid(0) = 0.5  # h_θ(x₁) = probabilité prédite
```

**Étape 3 : Calculer le gradient**
```python
# m = 3 (nombre d'exemples)

# Pour θ₀ (theta_0, le biais) - x₀=1 toujours pour tous les exemples
dJ_dtheta_0 = (1/m) × [(h_1 - y_1)×1 + (h_2 - y_2)×1 + (h_3 - y_3)×1]
            = (1/3) × [(0.5-1)×1 + (0.5-1)×1 + (0.5-0)×1]
            = (1/3) × [-0.5 - 0.5 + 0.5]
            = -0.167

# Pour θ₁ (theta_1, poids de la feature 1)
dJ_dtheta_1 = (1/m) × [(h_1 - y_1)×x₁₁ + (h_2 - y_2)×x₁₂ + (h_3 - y_3)×x₁₃]
            = (1/3) × [(0.5-1)×10 + (0.5-1)×12 + (0.5-0)×5]
            = (1/3) × [-5 - 6 + 2.5]
            = -2.833

# Pour θ₂ (theta_2, poids de la feature 2)
dJ_dtheta_2 = (1/m) × [(h_1 - y_1)×x₂₁ + (h_2 - y_2)×x₂₂ + (h_3 - y_3)×x₂₃]
            = (1/3) × [(0.5-1)×8 + (0.5-1)×9 + (0.5-0)×3]
            = (1/3) × [-4 - 4.5 + 1.5]
            = -2.333
```

**Étape 4 : Mettre à jour les paramètres**
```python
# alpha (α) = 0.01 (learning rate)

theta[0] = theta[0] - alpha × dJ_dtheta_0
         = 0 - 0.01 × (-0.167)
         = 0.00167

theta[1] = theta[1] - alpha × dJ_dtheta_1
         = 0 - 0.01 × (-2.833)
         = 0.02833

theta[2] = theta[2] - alpha × dJ_dtheta_2
         = 0 - 0.01 × (-2.333)
         = 0.02333
```

**Répéter pour plusieurs itérations jusqu'à convergence.**

---

### Points importants

#### 1. Normalisation des features

Avant l'entraînement, normaliser les features pour éviter que certaines dominent :
```python
x_normalized = (x - min) / (max - min)  # Entre 0 et 1
```

#### 2. Choix du learning rate

- **Trop petit** (ex: 0.0001) → convergence lente
- **Trop grand** (ex: 1.0) → peut diverger
- **Optimal** : généralement entre 0.001 et 0.1

#### 3. Nombre d'itérations

- **Trop peu** → modèle non entraîné
- **Trop** → risque de surapprentissage
- **Solution** : arrêter quand le coût ne diminue plus (convergence ou patience)

#### 4. Initialisation de θ

- Initialiser à zéro est souvent suffisant
- Initialisation aléatoire peut aider

---

### Checklist d'implémentation

- [ ] Implémenter la sigmoïde : `g(z) = 1/(1+e^(-z))`
- [ ] Implémenter la prédiction : `h_θ(x) = g(θᵀ · x)`
- [ ] Implémenter la fonction de coût : `J(θ) = -1/m × Σ[...]`
- [ ] Implémenter le gradient : `∂J/∂θⱼ = 1/m × Σ(h-y)×xⱼ`
- [ ] Implémenter la mise à jour : `θⱼ = θⱼ - α × gradient`
- [ ] Boucle d'entraînement avec plusieurs itérations
- [ ] Normaliser les features avant l'entraînement
- [ ] Implémenter la convergence ou la patience
- [ ] Tracer le coût pour vérifier la convergence

---

## ⏱️ Quand utiliser chaque formule

### Ordre d'exécution dans une epoch

```
POUR CHAQUE EPOCH :
  1. PRÉDIRE (h_θ)        ← Toujours d'abord
  2. MESURER (J)          ← Ensuite
  3. VÉRIFIER (convergence/patience) ← Décision
  4. CALCULER (gradient)  ← Pour ajuster
  5. AJUSTER (theta)      ← Dernière étape
```

**Mnémonique :** **P**rédire, **M**esurer, **V**érifier, **C**alculer, **A**juster

---

### Structure d'une epoch complète

```python
# 1. INITIALISATION (une fois)
# m = nombre d'exemples dans le dataset d'entraînement
# Exemple : si vous avez 1000 étudiants, m = 1000
# n_features = nombre de features (cours) sélectionnés
# Exemple : si vous avez sélectionné 3 cours, n_features = 3

theta = [0.0] * (n_features + 1)  # +1 pour le biais (θ₀)
best_cost = float('inf')
no_improvement = 0
previous_cost = float('inf')

# 2. BOUCLE D'ENTRAÎNEMENT
for epoch in range(max_epochs):
    
    # 2.1. CALCULER LES PRÉDICTIONS (h_θ(x))
    # Pour chaque exemple dans le dataset (m exemples au total)
    predictions = []
    for i in range(m):  # i va de 0 à m-1 (ex: 0, 1, 2, ..., 999 si m=1000)
        z = theta[0] * 1  # x₀ = 1 (biais)
        for j in range(n_features):
            z += theta[j+1] * x[i][j]
        h = sigmoid(z)
        predictions.append(h)
    
    # 2.2. CALCULER LE COÛT (J(θ))
    # ⚠️ IMPORTANT : y_binary doit être 0 ou 1 (pas y multiclasse 0,1,2,3)
    cost = 0.0
    for i in range(m):
        h = max(1e-15, min(1 - 1e-15, predictions[i]))  # Clipper pour éviter log(0)
        if y_binary[i] == 1:  # ⚠️ Utiliser y_binary, pas y multiclasse !
            cost += ln(h)
        else:  # y_binary[i] == 0
            cost += ln(1.0 - h)
    cost = -(1.0 / m) * cost
    
    # 2.3. VÉRIFIER CONVERGENCE
    if epoch > 0 and abs(previous_cost - cost) < convergence_threshold:
        print(f"✅ Convergence atteinte à l'epoch {epoch}")
        break
    
    # 2.4. VÉRIFIER PATIENCE
    if cost < best_cost:
        best_cost = cost
        best_theta = theta.copy()
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"⏹️  Arrêt (pas d'amélioration depuis {patience} epochs)")
            theta = best_theta  # Restaurer le meilleur
            break
    
    # 2.5. CALCULER LE GRADIENT (∂J/∂θⱼ)
    # ⚠️ IMPORTANT : y_binary doit être 0 ou 1 (pas y multiclasse 0,1,2,3)
    gradient = [0.0] * len(theta)
    for j in range(len(theta)):
        sum_error = 0.0
        for i in range(m):
            error = predictions[i] - y_binary[i]  # ⚠️ Utiliser y_binary, pas y multiclasse !
            x_ij = 1.0 if j == 0 else x[i][j-1]  # x₀=1 pour le biais
            sum_error += error * x_ij
        gradient[j] = (1.0 / m) * sum_error
    
    # 2.6. METTRE À JOUR THETA (θⱼ = θⱼ - α × gradient)
    for j in range(len(theta)):
        theta[j] = theta[j] - learning_rate * gradient[j]
    
    previous_cost = cost

return theta
```

---

### Tableau récapitulatif

| Formule | Quand l'utiliser | Fréquence | Obligatoire ? |
|---------|------------------|-----------|---------------|
| **h_θ(x) = g(θᵀ · x)** | Calculer les prédictions | Chaque epoch | ✅ Oui |
| **g(z) = 1/(1+e^(-z))** | Dans h_θ(x) | À chaque prédiction | ✅ Oui |
| **J(θ) = -1/m × Σ[...]** | Mesurer le coût | Chaque epoch | ✅ Oui (pour convergence/patience) |
| **∂J/∂θⱼ = 1/m × Σ(h-y)×xⱼ** | Calculer le gradient | Chaque epoch | ✅ Oui |
| **θⱼ = θⱼ - α × ∂J/∂θⱼ** | Mettre à jour theta | Chaque epoch | ✅ Oui |
| **\|coût_ancien - coût_nouveau\| < seuil** | Vérifier convergence | Chaque epoch | ⚠️ Optionnel |
| **no_improvement >= patience** | Early stopping | Chaque epoch | ⚠️ Optionnel |

---

## 🎯 Convergence et Patience

### Convergence

#### Définition

La convergence se produit quand le modèle a trouvé les paramètres optimaux. Le coût ne diminue plus significativement.

#### Signes de convergence

- Le coût J(θ) ne diminue plus (ou très peu)
- Les gradients sont proches de zéro
- Les paramètres ne changent plus beaucoup

#### Critères d'arrêt

- Nombre maximum d'itérations atteint
- Le coût ne diminue plus (seuil fixé, ex: 0.001)
- Les gradients sont très petits (ex: < 0.0001)

#### Implémentation

```python
convergence_threshold = 1e-6

if epoch > 0 and abs(previous_cost - current_cost) < convergence_threshold:
    print(f"✅ Convergence atteinte à l'epoch {epoch}")
    break
```

---

### Patience (Early Stopping)

#### Définition

La **patience** est le nombre d'epochs qu'on attend **sans amélioration** avant d'arrêter l'entraînement.

#### Analogie

Imaginez que vous cherchez un trésor dans une forêt :
- Vous marchez et trouvez parfois des indices (amélioration du coût)
- Si vous ne trouvez **rien de nouveau pendant 50 pas** (patience = 50), vous arrêtez
- Vous évitez de marcher indéfiniment sans rien trouver

#### Pourquoi utiliser la patience ?

**Avantages :**
1. **Évite le surapprentissage** : Arrête avant que le modèle mémorise les données
2. **Gain de temps** : Ne continue pas inutilement
3. **Meilleur modèle** : S'arrête au meilleur moment (quand le coût était le plus bas)

**Sans patience :**
```
Epoch 100: Coût = 0.5  ✅ Meilleur
Epoch 101-500: Coût = 0.5-0.6  ❌ Pas d'amélioration, continue quand même
Epoch 501-1000: Coût = 0.7-0.8  ❌ Dégradation (surapprentissage)
```

**Avec patience = 50 :**
```
Epoch 100: Coût = 0.5  ✅ Meilleur (sauvegardé)
Epoch 101-150: Coût = 0.5-0.6  ❌ Pas d'amélioration
Epoch 150: Arrêt ! (patience atteinte)
→ On garde le modèle de l'epoch 100 (meilleur)
```

#### Comment choisir la patience ?

- **Patience trop petite** (ex: 5) → Arrête trop vite, peut manquer des améliorations
- **Patience optimale** (ex: 50-100) → Bon équilibre
- **Patience trop grande** (ex: 1000) → Presque comme pas de patience

**Recommandation :** 50-100 epochs est généralement un bon compromis.

#### Différence avec convergence

- **Convergence** : Arrête quand le coût ne change **plus du tout** (variation < seuil)
- **Patience** : Arrête quand le coût n'**améliore plus** depuis N epochs

#### Implémentation complète

```python
max_epochs = 10000
patience = 50
best_cost = float('inf')
best_theta = None
no_improvement = 0

for epoch in range(max_epochs):
    # Entraînement
    current_cost = calculate_cost(theta, X, y)
    
    # Vérifier amélioration
    if current_cost < best_cost:
        best_cost = current_cost
        best_theta = theta.copy()  # Sauvegarder le meilleur modèle
        no_improvement = 0
        print(f"Epoch {epoch}: Nouveau meilleur coût = {best_cost:.6f}")
    else:
        no_improvement += 1
        
        # Early stopping si patience atteinte
        if no_improvement >= patience:
            print(f"⏹️  Arrêt à l'epoch {epoch} (pas d'amélioration depuis {patience} epochs)")
            print(f"✅ Meilleur coût atteint à l'epoch {epoch - patience}: {best_cost:.6f}")
            theta = best_theta  # Restaurer le meilleur modèle
            break
    
    # Mise à jour des paramètres
    gradient = calculate_gradient(theta, X, y)
    for j in range(len(theta)):
        theta[j] = theta[j] - learning_rate * gradient[j]

return theta
```

---

### Approche hybride recommandée

Utiliser un nombre maximum d'epochs comme limite de sécurité, avec un critère de convergence pour arrêter plus tôt, et la patience pour éviter le surapprentissage.

```python
max_epochs = 10000  # Limite de sécurité
convergence_threshold = 1e-6  # Seuil de convergence
patience = 50  # Arrêter si pas d'amélioration depuis 50 epochs

best_cost = float('inf')
best_theta = None
no_improvement = 0
previous_cost = float('inf')

for epoch in range(max_epochs):
    # Entraînement...
    current_cost = calculate_cost(...)
    
    # Vérifier convergence
    if epoch > 0 and abs(previous_cost - current_cost) < convergence_threshold:
        print(f"✅ Convergence à l'epoch {epoch}")
        break
    
    # Vérifier amélioration (patience)
    if current_cost < best_cost:
        best_cost = current_cost
        best_theta = theta.copy()
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"⏹️  Arrêt prématuré (pas d'amélioration depuis {patience} epochs)")
            theta = best_theta
            break
    
    previous_cost = current_cost
```

---

## 📁 Structure du projet

```
dslr/
├── data/
│   ├── dataset_train.csv      # Données d'entraînement
│   └── dataset_test.csv       # Données de test
├── srcs/
│   ├── describe.py            # Statistiques descriptives
│   ├── histogram.py          # Histogrammes et homogénéité
│   ├── logreg_train.py       # Entraînement régression logistique
│   ├── pair_plot.py          # Matrice de corrélation visuelle
│   ├── scatter_plot.py       # Graphiques de dispersion
│   └── utils.py              # Fonctions utilitaires
├── output/                   # Résultats et exports
├── cours_regression_logistique.md    # Cours complet (PDF disponible)
└── README.md                 # Ce fichier
```

---

## 💻 Utilisation

### Statistiques descriptives

```bash
python3 srcs/describe.py data/dataset_train.csv
```

### Analyse d'homogénéité

```bash
python3 srcs/histogram.py data/dataset_train.csv
```

### Entraînement du modèle de régression logistique

```bash
python3 srcs/logreg_train.py data/dataset_train.csv
```

### Visualisations

```bash
# Nuage de points (corrélations)
python3 srcs/scatter_plot.py data/dataset_train.csv

# Matrice de corrélation
python3 srcs/pair_plot.py data/dataset_train.csv
```

---

## 🔑 Points importants pour DSLR

### ⚠️ Structure des données : m = nombre TOTAL d'étudiants

**IMPORTANT** : Pour la régression logistique, vous avez besoin de :

1. **X** : Liste de listes, chaque ligne = un étudiant avec ses features
   ```python
   X = [
       [10, 8, 15, ...],  # Étudiant 1 : notes dans tous les cours
       [12, 9, 16, ...],  # Étudiant 2 : notes dans tous les cours
       [5, 3, 8, ...],    # Étudiant 3 : notes dans tous les cours
       ...
   ]
   ```

2. **y** : Liste de labels, chaque élément = la maison de l'étudiant correspondant
   ```python
   y = [0, 0, 1, 2, 3, 0, ...]  # 0=Gryffindor, 1=Hufflepuff, 2=Ravenclaw, 3=Slytherin
   ```

3. **m** : Nombre total d'étudiants
   ```python
   m = len(X)  # Exemple : si vous avez 1000 étudiants, m = 1000
   ```

**Exemple concret :**
- Si votre dataset a 1000 étudiants → **m = 1000**
- Chaque étudiant a 3 cours sélectionnés → **n_features = 3**
- X a 1000 lignes (une par étudiant)
- y a 1000 valeurs (une par étudiant)
- **m n'est PAS** le nombre de notes par cours, c'est le nombre d'étudiants !

### Transformation des données

Votre structure actuelle `houses_scores` est organisée par maison et par cours :
```python
houses_scores = {
    "Gryffindor": {
        "Arithmancy": [10, 8, 12, ...],  # Toutes les notes d'Arithmancy pour Gryffindor
        "Astronomy": [15, 14, 16, ...]
    },
    "Hufflepuff": {
        "Arithmancy": [9, 7, 11, ...],
        "Astronomy": [13, 12, 14, ...]
    },
    ...
}
```

**Il faut transformer cela en X et y :**

```python
# Pour chaque étudiant dans le dataset original :
# - Récupérer toutes ses notes (features)
# - Récupérer sa maison (label)
# - Ajouter à X et y

X = []  # Liste de listes
y = []  # Liste de labels

for i in range(nb_rows):  # Pour chaque ligne du CSV
    house = data[houses_index][i]
    features = []
    for course in courses_names:
        score = data[course_index][i]
        features.append(score)
    
    X.append(features)  # Ajouter les features de cet étudiant
    y.append(house_to_label[house])  # Ajouter le label de cet étudiant

m = len(X)  # Nombre total d'étudiants
```

### Sélection de features

Le projet sélectionne automatiquement les features les moins corrélées pour éviter la multicollinéarité :

```python
# Seuil de corrélation : 0.7
# Si |corrélation| > 0.7 entre deux features, on en garde qu'une
selected_features = utils.select_least_correlated_features(
    numeric_names, numeric_data,
    max_correlation=0.7,
    exclude_columns=["Index", "Hogwarts House"]
)
```

### Calcul de l'homogénéité

Le projet calcule l'homogénéité des cours pour identifier ceux qui discriminent le mieux les maisons :

```python
homogeneity = utils.calculate_homogeneity(houses_scores)
homogeneity_after_gap = utils.return_homogeneity_after_gap(homogeneity)
```

---

## 🎓 Pour aller plus loin

### Régression logistique multiclasse (One-vs-All)

#### ⚠️ Pourquoi One-vs-All ?

**La régression logistique est binaire par nature** (y ∈ {0, 1}). Pour classifier 4 maisons, on entraîne **4 modèles binaires séparés** :

- **Modèle 0** : Gryffindor vs (tous les autres)
- **Modèle 1** : Hufflepuff vs (tous les autres)
- **Modèle 2** : Ravenclaw vs (tous les autres)
- **Modèle 3** : Slytherin vs (tous les autres)

#### Conversion y multiclasse → y_binary

Pour chaque modèle, on convertit y multiclasse (0, 1, 2, 3) en y_binary (0 ou 1) :

```python
# Données originales
y = [0, 0, 1, 2, 3, 0]  # 0=Gryffindor, 1=Hufflepuff, 2=Ravenclaw, 3=Slytherin

# Pour le modèle Gryffindor (house_idx = 0)
y_binary = [1 if y[i] == 0 else 0 for i in range(m)]
# Résultat : [1, 1, 0, 0, 0, 1]

# Pour le modèle Hufflepuff (house_idx = 1)
y_binary = [1 if y[i] == 1 else 0 for i in range(m)]
# Résultat : [0, 0, 1, 0, 0, 0]
```

#### Structure du code

```python
# Entraîner 4 modèles séparés
all_thetas = []
houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

for house_idx, house_name in enumerate(houses):
    # Convertir y en binaire pour ce modèle
    y_binary = [1 if y[i] == house_idx else 0 for i in range(m)]
    
    # Entraîner le modèle binaire (utilise y_binary, pas y !)
    theta = logistic_regression_binary(X, y_binary, m, n_features)
    all_thetas.append(theta)
```

#### Prédiction

Pour prédire, on calcule la probabilité avec chaque modèle et on choisit la classe avec la probabilité la plus élevée.

### Régularisation

Pour éviter le surapprentissage, on peut ajouter un terme de régularisation :
```
J(θ) = -1/m × Σ[...] + λ/(2m) × Σ(θⱼ²)
```

Où :
- **λ (lambda)** = paramètre de régularisation (contrôle la force de la régularisation)
- Plus λ est grand → plus on pénalise les grands paramètres → modèle plus simple
- Plus λ est petit → moins de régularisation → modèle plus complexe

---

## 📝 Notes

- Toutes les fonctions mathématiques (sigmoid, exp, ln, etc.) sont implémentées from scratch
- Le projet utilise uniquement des bibliothèques Python standard (csv, typing)
- Les visualisations utilisent matplotlib

---

**Bon courage pour l'implémentation ! 🚀**

---

## 🧠 Comprendre `srcs/logreg_train.py` (explication guidée)

Cette section explique **ce que fait le fichier actuellement**. Important: l’**entraînement est désactivé** pour te permettre de le réimplémenter proprement (one-vs-all).

### Imports

- **`import utils`**: importe `srcs/utils.py` (chargement CSV, sélection de features, maths, corrélations…).
- **`import sys`**: permet de lire `sys.argv` (argument `dataset_train.csv`).
- **`from typing ...`**: annotations de types (aide à la lisibilité, pas indispensable à l’exécution).

### `logreg_train()` (la fonction “main”)

Cette fonction orchestre tout le pipeline “data prep”.

- **Vérification des arguments**
	- attend exactement 1 argument: le chemin vers le dataset.
- **Chargement du CSV**
	- `utils.load_csv(..., return_header=True, auto_convert=True)` renvoie:
		- `header`: noms de colonnes
		- `data`: valeurs **par colonnes**
- **Extraction des colonnes numériques**
	- `utils.extract_numeric_columns(...)` renvoie `numeric_names` / `numeric_data`.
- **Sélection de features peu corrélées**
	- `utils.select_least_correlated_features(..., max_correlation=0.7, exclude_columns=["Index", "Hogwarts House"])`
	- résultat: `selected_features`
	- ensuite `courses_names` retire “Index” pour ne garder que les cours.
- **Préparation du dataset d’entraînement**
	- `X, y, m = prepare_training_data(header, data, courses_names)`
	- affichage d’un exemple (`X[0]`, `y[0]`) pour vérifier.
- **Entraînement**
	- actuellement **désactivé**: le script affiche juste un message et s’arrête.

### `prepare_training_data(header, data, features_names) -> (X, y, m)`

- **But**: convertir le CSV (par colonnes) en format ML standard:
	- **`X`**: liste de lignes, 1 ligne = 1 étudiant, colonnes = notes des cours choisis
	- **`y`**: label multiclasse (0..3) correspondant à la maison
	- **`m`**: nombre d’exemples gardés
- **Pourquoi “par étudiant” et pas “par maison/cours”**
	- L’entraînement d’une régression logistique (et les formules coût/gradient) suppose des exemples \(x^{(i)}\) complets: un **vecteur de features** pour un individu.
	- Si tu regroupes “maison → cours → liste de notes”, tu perds l’alignement “ces notes appartiennent au **même étudiant**”, donc tu ne peux plus former des vecteurs \(x = [\text{Astronomy}, \text{Potions}, ...]\) cohérents.
	- Le format “par maison” est utile pour l’analyse/visualisation (comparer des distributions), mais pas pour entraîner un modèle multi-features.
- **Détails importants**
	- construit `house_to_label` en triant les maisons (`sorted(...)`), puis en les numérotant.
	- construit `features_indices` pour accéder vite aux colonnes des cours.
	- parcourt chaque étudiant `i`:
		- si la maison est `None` → skip
		- si une feature est `None` → skip la ligne complète (exemple ignoré)
		- sinon ajoute la ligne dans `X` et le label dans `y`.

### `if __name__ == "__main__":`

- si tu lances `python3 srcs/logreg_train.py ...`, Python appelle `logreg_train()`.

### Ce que tu coderas ensuite (rappel)

- **L’entraînement doit être one-vs-all**:
	- pour chaque maison `k`:
		- `y_binary = [1 if yi == k else 0 for yi in y]`
		- entraîner un modèle binaire → produire un `theta_k`
	- sauvegarder les 4 `theta_k` pour `logreg_predict`.

---

## 🧰 Comprendre `srcs/utils.py` (explication guidée, par blocs)

`srcs/utils.py` est une “boîte à outils” utilisée par tout le projet: lecture CSV, nettoyage, stats “describe”, corrélations/feature selection, maths (exp/ln/sigmoid), et quelques helpers (encore incomplets) pour la logreg.

Pour rester lisible, on suit l’ordre du fichier **de haut en bas** et on explique ce que fait chaque bloc/fonction.

### 1) Imports

- **`csv`**: lecture/écriture CSV.
- **`typing`**: annotations de types.
- **`datetime`**: parsing possible de dates lors de `auto_convert`.
- **`matplotlib.pyplot as plt`**: utilisé par les fonctions de plots (scatter, etc.).

### 2) Chargement CSV et inférence de types

#### `load_csv(path, columns=None, skip_header=True, convert_type=float, auto_convert=False, parse_dates=True, return_header=False)`

- **But**: lire un CSV et renvoyer les données au format **par colonnes** (transposé).
- **Points importants**
	- si `return_header=True`, la fonction lit la 1ère ligne pour construire `header`.
	- si `skip_header=True` et `return_header=False`, elle saute la 1ère ligne.
	- si `columns` est fourni, elle filtre lignes + header sur ces indices.
	- conversion:
		- `auto_convert=True` → `_auto_convert_value` (int/float/None/date/str)
		- sinon `convert_type` (par défaut `float`) → conversion simple
	- à la fin, elle transpose: `zip(*data)` pour obtenir `List[List]` où chaque sous-liste est une colonne.

#### `_auto_convert_value(value, parse_dates=True)`

- **But**: convertir une cellule string en type “le plus probable”.
- **Règles**
	- vide / `nan` / `null` / `none` / `n/a` → `None`
	- sinon tente `int` (avec des heuristiques)
	- sinon tente `float`
	- sinon tente `datetime` si `parse_dates=True`
	- sinon renvoie la string originale

#### `analyze_csv_types(path, sample_size=100)`

- **But**: inspecter un échantillon de lignes et compter les types détectés par `_auto_convert_value`.
- **Sortie**: pour chaque colonne: `dominant_type`, % de null, 3 exemples bruts, distribution des types.

#### `get_numeric_columns(path, skip_header=True)`

- **But**: retourner `(numeric_names, numeric_indices)` des colonnes dont le type dominant est `int` ou `float`.
- **Utilisé ensuite** par `extract_numeric_columns`.

#### `extract_numeric_columns(path, data, skip_header=True)`

- **But**: filtrer les données (chargées via `load_csv`) pour ne garder que les colonnes numériques.
- **Entrée `data`**: peut être soit:
	- `columns_data` (juste les colonnes)
	- ou `(header, columns_data)` (si tu as demandé le header)
- **Sortie**: `(numeric_names, numeric_data)` au format **par colonnes**.

#### `none_filter(data)`

- **But**: enlever les `None` de chaque colonne d’une liste de colonnes.
- **Attention**: ça casse l’alignement “ligne” entre colonnes (utile pour histogrammes, moins pour corrélations/paires).

### 3) Valeurs manquantes

#### `count_missing_values(path, skip_header=True)`

- **But**: compter, pour chaque colonne, le nombre de cellules vides vs pleines.
- Reconnaît vides: `''` ou `nan/null/none/n/a` (case-insensitive).
- **Sortie**: dict avec stats par colonne + globales.

#### `print_missing_values_report(path, skip_header=True, show_all_columns=False)`

- **But**: afficher joliment les résultats de `count_missing_values`.

### 4) Sauvegarde/chargement de “params”

#### `save_model_params(params, path)`

- **But**: écrire une liste de nombres dans un fichier, séparés par des espaces.
- **Typiquement**: sauvegarder un vecteur `theta`.

#### `load_model_params(path, expected_count=None)`

- **But**: relire le fichier et le convertir en `List[float]`.
- Si `expected_count` est fourni, vérifie la taille.

### 5) Min/Max/normalisation/standardisation

#### `ft_min`, `ft_max`, `get_min_max`

- **But**: implémenter min/max sans utiliser `min()`/`max()` directement (utile pour rester “from scratch”).

#### `normalize_min_max(data, min_val=None, max_val=None)`

- **But**: normaliser entre 0 et 1: \((x - min) / (max - min)\).
- Si `max == min`, renvoie `0.5` partout (évite division par 0).

#### `standardize(data)`

- **But**: standardiser: \((x - \mu)/\sigma\).
- Renvoie `(data_standardized, mean, std)`.

### 6) Fonctions de plot (utilisées par les scripts de visualisation)

#### `plot_scatter(x_data, y_data, output_path, ..., regression_line=None, norm_params=None, figsize=(12, 8))`

- **But**: scatter plot, et optionnellement tracer une droite de régression.
- Si `norm_params` est fourni, normalise la plage X avant de calculer la droite.
- Sauvegarde un PNG + `plt.show()`.

#### `calculate_r2`, `calculate_mse`, `calculate_mae`, `linear_prediction`

- **But**: métriques / prédictions associées (surtout utiles pour la partie régression linéaire / plots).

### 7) Statistiques “describe”

#### `count(data)`

- **But**: renvoyer `(nb_colonnes, [len(col) ...])`.

#### `mean`, `std`

- **But**: moyenne et écart-type (population, pas “échantillon”).

#### `median`, `percentile`, `quartiles`

- **But**: quantiles “from scratch” via tri + interpolation linéaire.
- `quartiles` appelle `percentile(25/50/75)`.

### 8) Corrélations et sélection de features

#### `correlation_coefficient(x, y)`

- **But**: Pearson r = covariance / (std_x * std_y).
- **Attention**: suppose `x` et `y` alignés et de même longueur.

#### `compute_correlation_matrix(column_names, column_data, exclude_columns=None)`

- **But**: calculer une matrice `dict[name1][name2] = corr`.
- Filtre les paires en retirant les indices où l’une des deux colonnes vaut `None`.

#### `select_least_correlated_features(column_names, column_data, max_correlation=0.8, exclude_columns=None)`

- **But**: garder une liste de features en évitant d’ajouter une feature trop corrélée à une déjà sélectionnée.
- Stratégie simple: on parcourt `candidate_names` et on compare à `selected_features` déjà gardées.

#### `print_correlation_matrix(...)`

- **But**: afficher la matrice (complète ou seulement les fortes), avec un résumé.

### 9) Maths “logreg”

#### `exp(x, precision=50)`

- **But**: approx de \(e^x\) via série de Taylor.
- Protection contre overflow: refuse `x > 700` / `x < -700`.

#### `ln(x, precision=50)` + `_ln_small(x, precision=50)`

- **But**: approx de \(\ln(x)\).
- Utilise:
	- réduction si `x >= 2` (divisions par 2 + `ln(2)` précalculé)
	- transformation si `x < 0.5` (\(\ln(x) = -\ln(1/x)\))
	- sinon série de Taylor autour de 1 (`_ln_small`)

#### `sigmoid(value)`

- **But**: \(1/(1+e^{-value})\) en utilisant `exp`.

### 10) Homogénéité (analyse)

#### `calculate_homogeneity(scores_by_class)` / `return_homogeneity_after_gap(...)` / `find_gap(...)`

- **But**: calculer une métrique d’homogénéité et sélectionner des cours “au-dessus d’un gap”.
- Utilisé pour l’analyse (ex: histogrammes), pas requis pour l’entraînement logreg.

### 11) Helpers logreg en bas de fichier (à manier avec prudence)

Ces fonctions existent mais ne sont pas toutes “prêtes”:

- **`calculate_cost_function(...)`**
	- écrit la log-loss binaire mais prend `y` décrit comme multiclasse (0..3) → **à n’utiliser que avec `y_binary`**, sinon c’est faux.
- **`transform_ybinary(y)`**
	- transforme seulement “classe 0 vs reste” (codé en dur) → pour one-vs-all il faut une version `transform_ybinary(y, k)`.
- **`calculate_gradient(...)`**
	- retourne actuellement un gradient nul (placeholder).
- **`calculate_prediction(...)`**
	- calcule bien `h_theta(x)` (sigmoid du produit \(\theta^T x\) avec biais), utile quand tu recoderas l’entraînement.
