# 📋 Quand utiliser les différentes formules de la régression logistique

## **🔄 Flux complet de l'entraînement**

Voici l'ordre d'exécution et quand utiliser chaque formule :

---

## **ÉTAPE 1 : Initialisation (avant la boucle)**

### **Formule utilisée : Aucune formule mathématique**

```python
# Initialiser theta à zéro
theta = [0.0] * (n_features + 1)  # +1 pour le biais

# Paramètres
learning_rate = 0.01
max_epochs = 10000
patience = 50
```

**Quand :** Une seule fois, au début de l'entraînement

---

## **ÉTAPE 2 : Boucle d'entraînement (pour chaque epoch)**

### **2.1. Calculer les prédictions (h_θ(x))**

**Formule utilisée :**
```
h_θ(x) = g(θᵀ · x)
```

**Où :**
- `g(z) = 1/(1+e^(-z))` = sigmoïde
- `θᵀ · x = θ₀×1 + θ₁×x₁ + θ₂×x₂ + ...`

**Quand :** À chaque epoch, pour TOUS les exemples

**Code :**
```python
for i in range(m):  # Pour chaque exemple
    # Calculer z = θᵀ · x
    z = theta[0] * 1  # x₀ = 1 (biais)
    for j in range(len(features)):
        z += theta[j+1] * x[i][j]
    
    # Appliquer sigmoïde
    h = utils.sigmoid(z)
    predictions.append(h)
```

**Pourquoi :** On a besoin des prédictions pour calculer le coût et le gradient

---

### **2.2. Calculer le coût (J(θ))**

**Formule utilisée :**
```
J(θ) = -1/m × Σ[i=1 à m] [yᵢ × log(h_θ(xᵢ)) + (1-yᵢ) × log(1-h_θ(xᵢ))]
```

**Quand :** 
- À chaque epoch (optionnel, pour suivre la progression)
- Ou seulement tous les N epochs (ex: tous les 100 epochs)
- **OBLIGATOIRE** pour vérifier la convergence et la patience

**Code :**
```python
cost = 0.0
for i in range(m):
    h = predictions[i]
    y = y_true[i]
    
    # Clipper h pour éviter log(0)
    h = max(1e-15, min(1 - 1e-15, h))
    
    if y == 1:
        cost += utils.ln(h)
    else:
        cost += utils.ln(1.0 - h)

cost = -(1.0 / m) * cost
```

**Pourquoi :** 
- Mesurer la performance du modèle
- Détecter la convergence
- Détecter l'absence d'amélioration (patience)

---

### **2.3. Calculer le gradient (∂J/∂θⱼ)**

**Formule utilisée :**
```
∂J/∂θⱼ = 1/m × Σ[i=1 à m] (h_θ(xᵢ) - yᵢ) × xᵢⱼ
```

**Quand :** À chaque epoch, pour CHAQUE paramètre θⱼ

**Code :**
```python
gradient = [0.0] * len(theta)

for j in range(len(theta)):  # Pour chaque paramètre
    sum_error = 0.0
    for i in range(m):  # Pour chaque exemple
        error = predictions[i] - y_true[i]
        x_ij = 1.0 if j == 0 else x[i][j-1]  # x₀=1 pour le biais
        sum_error += error * x_ij
    
    gradient[j] = (1.0 / m) * sum_error
```

**Pourquoi :** Le gradient indique comment ajuster chaque paramètre

---

### **2.4. Mettre à jour les paramètres (θⱼ)**

**Formule utilisée :**
```
θⱼ = θⱼ - α × ∂J/∂θⱼ
```

**Quand :** À chaque epoch, après avoir calculé le gradient

**Code :**
```python
for j in range(len(theta)):
    theta[j] = theta[j] - learning_rate * gradient[j]
```

**Pourquoi :** C'est l'étape qui fait "apprendre" le modèle

---

### **2.5. Vérifier la convergence**

**Formule utilisée :**
```
|coût_ancien - coût_nouveau| < seuil
```

**Quand :** À chaque epoch, après avoir calculé le coût

**Code :**
```python
if abs(previous_cost - current_cost) < convergence_threshold:
    print(f"✅ Convergence atteinte à l'epoch {epoch}")
    break
```

**Pourquoi :** Arrêter quand le modèle ne s'améliore plus

---

### **2.6. Vérifier la patience (early stopping)**

**Formule utilisée :**
```
Si coût_nouveau >= meilleur_coût:
    no_improvement += 1
Si no_improvement >= patience:
    arrêter
```

**Quand :** À chaque epoch, après avoir calculé le coût

**Code :**
```python
if current_cost < best_cost:
    best_cost = current_cost
    best_theta = theta.copy()
    no_improvement = 0
else:
    no_improvement += 1
    if no_improvement >= patience:
        print(f"⏹️  Arrêt (pas d'amélioration depuis {patience} epochs)")
        break
```

**Pourquoi :** Éviter le surapprentissage en arrêtant tôt

---

## **📊 Résumé : Ordre d'exécution complet**

```python
# 1. INITIALISATION (une fois)
theta = [0.0] * (n_features + 1)
best_cost = float('inf')
no_improvement = 0

# 2. BOUCLE D'ENTRAÎNEMENT
for epoch in range(max_epochs):
    
    # 2.1. CALCULER LES PRÉDICTIONS (h_θ(x))
    predictions = []
    for i in range(m):
        z = theta[0] * 1 + sum(theta[j+1] * x[i][j] for j in range(n_features))
        h = sigmoid(z)
        predictions.append(h)
    
    # 2.2. CALCULER LE COÛT (J(θ))
    cost = calculate_cost(predictions, y_true)
    
    # 2.3. VÉRIFIER CONVERGENCE
    if epoch > 0 and abs(previous_cost - cost) < convergence_threshold:
        break
    
    # 2.4. VÉRIFIER PATIENCE
    if cost < best_cost:
        best_cost = cost
        best_theta = theta.copy()
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= patience:
            theta = best_theta  # Restaurer le meilleur
            break
    
    # 2.5. CALCULER LE GRADIENT (∂J/∂θⱼ)
    gradient = calculate_gradient(predictions, y_true, x)
    
    # 2.6. METTRE À JOUR THETA (θⱼ = θⱼ - α × gradient)
    for j in range(len(theta)):
        theta[j] = theta[j] - learning_rate * gradient[j]
    
    previous_cost = cost

return theta
```

---

## **📝 Tableau récapitulatif**

| Formule | Quand l'utiliser | Fréquence | Obligatoire ? |
|---------|------------------|-----------|---------------|
| **h_θ(x) = g(θᵀ · x)** | Calculer les prédictions | Chaque epoch | ✅ Oui |
| **g(z) = 1/(1+e^(-z))** | Dans h_θ(x) | À chaque prédiction | ✅ Oui |
| **J(θ) = -1/m × Σ[...]** | Mesurer le coût | Chaque epoch (ou tous les N) | ✅ Oui (pour convergence/patience) |
| **∂J/∂θⱼ = 1/m × Σ(h-y)×xⱼ** | Calculer le gradient | Chaque epoch | ✅ Oui |
| **θⱼ = θⱼ - α × ∂J/∂θⱼ** | Mettre à jour theta | Chaque epoch | ✅ Oui |
| **|coût_ancien - coût_nouveau| < seuil** | Vérifier convergence | Chaque epoch | ⚠️ Optionnel |
| **no_improvement >= patience** | Early stopping | Chaque epoch | ⚠️ Optionnel |

---

## **🎯 Exemple concret étape par étape**

**Epoch 1 :**

```python
# 1. Prédictions
z = theta[0]*1 + theta[1]*10 + theta[2]*8  # = 0 (car theta initialisé à 0)
h = sigmoid(0) = 0.5  # Pour chaque exemple

# 2. Coût
J = -1/m × [y×log(0.5) + (1-y)×log(0.5)]  # ≈ 0.693 (mauvais)

# 3. Gradient
∂J/∂θ₀ = 1/m × Σ(0.5 - y) × 1
∂J/∂θ₁ = 1/m × Σ(0.5 - y) × x₁
∂J/∂θ₂ = 1/m × Σ(0.5 - y) × x₂

# 4. Mise à jour
theta[0] = 0 - 0.01 × gradient[0]
theta[1] = 0 - 0.01 × gradient[1]
theta[2] = 0 - 0.01 × gradient[2]

# 5. Vérifications
# Convergence ? Non (première itération)
# Patience ? Pas encore (première itération)
```

**Epoch 2 :**

```python
# 1. Prédictions (avec nouveaux theta)
z = theta[0]*1 + theta[1]*10 + theta[2]*8  # ≠ 0 maintenant
h = sigmoid(z)  # Probablement différent de 0.5

# 2. Coût
J = ...  # Probablement meilleur qu'epoch 1

# 3. Vérifier amélioration
if J < best_cost:
    best_cost = J
    no_improvement = 0
else:
    no_improvement = 1  # Commence à compter

# 4. Gradient et mise à jour (même processus)
```

**Epoch N (convergence) :**

```python
# 1-4. Même processus

# 5. Vérification convergence
if abs(previous_cost - current_cost) < 1e-6:
    print("✅ Convergence atteinte !")
    break  # Arrêter ici
```

---

## **🔑 Points clés**

1. **Prédictions (h_θ)** → TOUJOURS en premier dans chaque epoch
2. **Coût (J)** → Calculé après les prédictions, utilisé pour vérifier convergence/patience
3. **Gradient** → Calculé après le coût, nécessaire pour la mise à jour
4. **Mise à jour** → Dernière étape de chaque epoch
5. **Vérifications** → Après calcul du coût, pour décider si on continue

---

## **💡 Astuce : Ordre à retenir**

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

**Maintenant vous savez quand utiliser chaque formule ! 🎓**

