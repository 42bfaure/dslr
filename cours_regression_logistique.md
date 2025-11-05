# 📚 Cours complet : Mathématiques de la régression logistique

## **1. Objectif et contexte**

### **Problème à résoudre :**
On veut faire de la **classification binaire** : prédire si un exemple appartient à la classe 0 ou 1.

**Exemple :** Prédire si un étudiant appartient à "Gryffindor" (y=1) ou non (y=0) à partir de ses notes.

### **Différence avec la régression linéaire :**
- **Régression linéaire** : Sortie continue (ex: prix = 1000 + 50×taille)
- **Régression logistique** : Sortie = probabilité entre 0 et 1

---

## **2. La fonction sigmoïde**

### **Définition :**
```
g(z) = 1 / (1 + e^(-z))
```

Où `e` est le nombre d'Euler (≈ 2.718).

### **Propriétés :**
- `g(z)` est toujours entre 0 et 1
- Si z → -∞ → g(z) → 0
- Si z = 0 → g(z) = 0.5
- Si z → +∞ → g(z) → 1

### **Forme de la courbe :**
```
g(z)
 1 |     ╱────────────
   |    ╱
0.5|───╱
   | ╱
 0 |╱
   └───────────→ z
```

### **Pourquoi la sigmoïde ?**
Elle transforme une combinaison linéaire (qui peut être n'importe quel nombre) en une probabilité (toujours entre 0 et 1).

---

## **3. Le modèle : h_θ(x)**

### **Formule :**
```
h_θ(x) = g(θᵀ · x)
```

Où :
- `θ` = vecteur de paramètres `[θ₀, θ₁, θ₂, ..., θₙ]`
- `x` = vecteur de features `[x₀=1, x₁, x₂, ..., xₙ]` (x₀=1 pour le biais)
- `θᵀ · x` = produit scalaire = `θ₀×1 + θ₁×x₁ + θ₂×x₂ + ... + θₙ×xₙ`

### **Exemple concret :**
```python
θ = [0.5, 0.3, -0.2]  # [θ₀, θ₁, θ₂]
x = [1, 10, 8]        # [x₀=1, x₁=10, x₂=8]

# Calculer θᵀ · x
z = θ₀×1 + θ₁×10 + θ₂×8
  = 0.5×1 + 0.3×10 + (-0.2)×8
  = 0.5 + 3 - 1.6
  = 1.9

# Appliquer la sigmoïde
h_θ(x) = g(1.9) = 1/(1+e^(-1.9)) ≈ 0.87

→ Probabilité de 87% que cet étudiant soit dans Gryffindor
```

---

## **4. La fonction de coût (Loss Function)**

### **Formule :**
```
J(θ) = -1/m × Σ[i=1 à m] [yᵢ × log(h_θ(xᵢ)) + (1-yᵢ) × log(1-h_θ(xᵢ))]
```

Où :
- `m` = nombre d'exemples d'entraînement
- `yᵢ` = label réel (0 ou 1) pour l'exemple i
- `h_θ(xᵢ)` = probabilité prédite pour l'exemple i
- `log` = logarithme naturel (ln)

### **Pourquoi cette formule ?**

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

### **Visualisation :**
```
Coût
 5 |     ╱
   |    ╱
   |   ╱
 2 |  ╱
   | ╱
 0 |╱───────
   └───────→ h_θ(x)
   0       1
```

---

## **5. La dérivée partielle (gradient)**

### **Formule :**
```
∂J/∂θⱼ = 1/m × Σ[i=1 à m] (h_θ(xᵢ) - yᵢ) × xᵢⱼ
```

Où :
- `j` = indice du paramètre (0, 1, 2, ..., n)
- `xᵢⱼ` = valeur de la feature j pour l'exemple i

### **Interprétation :**

Cette formule nous dit **comment ajuster chaque paramètre θⱼ** pour réduire le coût.

**Détails :**
- `(h_θ(xᵢ) - yᵢ)` = erreur de prédiction
  - Si h > y → erreur positive → on doit diminuer θⱼ
  - Si h < y → erreur négative → on doit augmenter θⱼ
- `xᵢⱼ` = poids de la feature
  - Si x est grand → ajustement plus important
  - Si x est petit → ajustement plus faible

### **Exemple :**
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

## **6. Algorithme de descente de gradient**

### **Principe :**

On met à jour chaque paramètre dans la direction opposée au gradient pour réduire le coût.

### **Formule de mise à jour :**
```
θⱼ = θⱼ - α × ∂J/∂θⱼ
```

Où :
- `α` = learning rate (taux d'apprentissage), ex: 0.01

### **Algorithme complet :**

```python
1. Initialiser θ = [0, 0, 0, ..., 0]

2. Pour chaque itération :
   a) Calculer h_θ(x) pour tous les exemples
   b) Calculer le gradient ∂J/∂θⱼ pour chaque paramètre
   c) Mettre à jour : θⱼ = θⱼ - α × ∂J/∂θⱼ
   d) (Optionnel) Calculer J(θ) pour suivre la progression

3. Répéter jusqu'à convergence
```

### **Exemple étape par étape :**

**Itération 1 :**
```python
θ = [0, 0, 0]  # Initialisation

# Exemple 1: x=[1, 10, 8], y=1
z = 0×1 + 0×10 + 0×8 = 0
h = sigmoid(0) = 0.5
Erreur = 0.5 - 1 = -0.5

# Gradient
∂J/∂θ₀ = -0.5 × 1 = -0.5
∂J/∂θ₁ = -0.5 × 10 = -5
∂J/∂θ₂ = -0.5 × 8 = -4

# Mise à jour (α=0.01)
θ₀ = 0 - 0.01 × (-0.5) = 0.005
θ₁ = 0 - 0.01 × (-5) = 0.05
θ₂ = 0 - 0.01 × (-4) = 0.04
```

**Itération 2 :**
```python
θ = [0.005, 0.05, 0.04]

# Exemple 1: x=[1, 10, 8], y=1
z = 0.005×1 + 0.05×10 + 0.04×8 = 0.005 + 0.5 + 0.32 = 0.825
h = sigmoid(0.825) ≈ 0.695
Erreur = 0.695 - 1 = -0.305

# Gradient (plus petit qu'avant, on progresse!)
∂J/∂θ₀ = -0.305 × 1 = -0.305
∂J/∂θ₁ = -0.305 × 10 = -3.05
∂J/∂θ₂ = -0.305 × 8 = -2.44

# Mise à jour
θ₀ = 0.005 - 0.01 × (-0.305) = 0.00805
θ₁ = 0.05 - 0.01 × (-3.05) = 0.0805
θ₂ = 0.04 - 0.01 × (-2.44) = 0.0644
```

À chaque itération, le modèle s'améliore !

---

## **7. Dérivation mathématique du gradient**

### **Étape 1 : Décomposer la fonction de coût**

```
J(θ) = -1/m × Σ [y × log(h) + (1-y) × log(1-h)]
```

Où `h = h_θ(x) = g(z)` et `z = θᵀ · x`.

### **Étape 2 : Dériver par rapport à θⱼ**

On utilise la **règle de la chaîne** :
```
∂J/∂θⱼ = ∂J/∂h × ∂h/∂z × ∂z/∂θⱼ
```

### **Étape 3 : Calculer chaque terme**

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

### **Étape 4 : Combiner**

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

## **8. Points importants**

### **1. Normalisation des features**
Avant l'entraînement, normaliser les features pour éviter que certaines dominent :
```python
x_normalized = (x - min) / (max - min)  # Entre 0 et 1
```

### **2. Choix du learning rate**
- **Trop petit** (ex: 0.0001) → convergence lente
- **Trop grand** (ex: 1.0) → peut diverger
- **Optimal** : généralement entre 0.001 et 0.1

### **3. Nombre d'itérations**
- **Trop peu** → modèle non entraîné
- **Trop** → risque de surapprentissage
- **Solution** : arrêter quand le coût ne diminue plus

### **4. Initialisation de θ**
- Initialiser à zéro est souvent suffisant
- Initialisation aléatoire peut aider

---

## **9. Résumé des formules clés**

```
1. Sigmoïde :          g(z) = 1/(1+e^(-z))

2. Prédiction :        h_θ(x) = g(θᵀ · x)

3. Fonction de coût :  J(θ) = -1/m × Σ[y×log(h) + (1-y)×log(1-h)]

4. Gradient :          ∂J/∂θⱼ = 1/m × Σ(h - y) × xⱼ

5. Mise à jour :       θⱼ = θⱼ - α × ∂J/∂θⱼ
```

---

## **10. Exemple complet simplifié**

**Données :**
```python
X = [[10, 8], [12, 9], [5, 3]]  # 3 exemples, 2 features
y = [1, 1, 0]                    # Labels
```

**Étape 1 : Initialiser**
```python
θ = [0, 0, 0]  # [θ₀, θ₁, θ₂]
α = 0.01
```

**Étape 2 : Calculer les prédictions**
```python
# Exemple 1: x=[1, 10, 8]
z = 0×1 + 0×10 + 0×8 = 0
h₁ = sigmoid(0) = 0.5

# Exemple 2: x=[1, 12, 9]
z = 0
h₂ = sigmoid(0) = 0.5

# Exemple 3: x=[1, 5, 3]
z = 0
h₃ = sigmoid(0) = 0.5
```

**Étape 3 : Calculer le gradient**
```python
# Pour θ₀ (x₀=1 toujours)
∂J/∂θ₀ = 1/3 × [(0.5-1)×1 + (0.5-1)×1 + (0.5-0)×1]
       = 1/3 × [-0.5 - 0.5 + 0.5]
       = -0.167

# Pour θ₁
∂J/∂θ₁ = 1/3 × [(0.5-1)×10 + (0.5-1)×12 + (0.5-0)×5]
       = 1/3 × [-5 - 6 + 2.5]
       = -2.833

# Pour θ₂
∂J/∂θ₂ = 1/3 × [(0.5-1)×8 + (0.5-1)×9 + (0.5-0)×3]
       = 1/3 × [-4 - 4.5 + 1.5]
       = -2.333
```

**Étape 4 : Mettre à jour**
```python
θ₀ = 0 - 0.01 × (-0.167) = 0.00167
θ₁ = 0 - 0.01 × (-2.833) = 0.02833
θ₂ = 0 - 0.01 × (-2.333) = 0.02333
```

**Répéter pour plusieurs itérations jusqu'à convergence.**

---

## **11. Checklist d'implémentation**

- [ ] Implémenter la sigmoïde : `g(z) = 1/(1+e^(-z))`
- [ ] Implémenter la prédiction : `h_θ(x) = g(θᵀ · x)`
- [ ] Implémenter la fonction de coût : `J(θ) = -1/m × Σ[...]`
- [ ] Implémenter le gradient : `∂J/∂θⱼ = 1/m × Σ(h-y)×xⱼ`
- [ ] Implémenter la mise à jour : `θⱼ = θⱼ - α × gradient`
- [ ] Boucle d'entraînement avec plusieurs itérations
- [ ] Normaliser les features avant l'entraînement
- [ ] Tracer le coût pour vérifier la convergence

---

## **12. Pour aller plus loin**

### **Régression logistique multiclasse (One-vs-All)**

Pour classifier plusieurs classes (ex: 4 maisons de Poudlard), on entraîne un modèle par classe :
- Modèle 1 : Gryffindor vs (tous les autres)
- Modèle 2 : Hufflepuff vs (tous les autres)
- Modèle 3 : Ravenclaw vs (tous les autres)
- Modèle 4 : Slytherin vs (tous les autres)

Pour prédire, on choisit la classe avec la probabilité la plus élevée.

### **Régularisation**

Pour éviter le surapprentissage, on peut ajouter un terme de régularisation :
```
J(θ) = -1/m × Σ[...] + λ/(2m) × Σ(θⱼ²)
```

Où λ (lambda) contrôle la force de la régularisation.

---

**Bon courage pour l'implémentation ! 🚀**

