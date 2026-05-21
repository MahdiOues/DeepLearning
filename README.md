# 🍎🥭 Fruit & Vegetable Disease Classifier

> Détection automatique de maladies sur fruits et légumes par Deep Learning (ResNet50V2)

---

## Table des matières
- [Présentation du problème](#problème)
- [Jeu de données](#jeu-de-données)
- [Architecture choisie](#architecture)
- [Performances](#performances)
- [Structure du projet](#structure)
- [Installation & Exécution](#exécution)
- [Résultats des expériences](#expériences)

---

## Problème

La détection précoce des maladies sur les fruits et légumes est un enjeu majeur pour réduire les pertes alimentaires et les traitements phytosanitaires. Ce projet entraîne un classifieur d'images capable de distinguer **fruits/légumes sains vs pourris** pour 4 espèces courantes.

**Tâche** : classification multi-classes (8 classes : 4 espèces × 2 états)  
**Entrée** : image RGB d'un fruit/légume  
**Sortie** : classe prédite + score de confiance

---

## Jeu de données

| Propriété | Valeur |
|-----------|--------|
| Source | [Fruit and Vegetable Disease — Kaggle](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten) |
| Taille totale | ~5 GB, 28 classes originales |
| Classes retenues | 8 classes (Apple, Banana, Orange, Mango × Healthy/Rotten) |
| Split | 70% train / 20% validation / 10% test |

**Classes** :
```
Apple__Healthy    Apple__Rotten
Banana__Healthy   Banana__Rotten
Orange__Healthy   Orange__Rotten
Mango__Healthy    Mango__Rotten
```

**Prétraitement** :
- Suppression des fichiers corrompus (double vérification PIL + TensorFlow)
- Redimensionnement à 128×128 pixels
- Normalisation ResNet (`preprocess_input`) → valeurs dans [-1, 1]
- Poids de classe automatiques (`compute_class_weight='balanced'`)

**Augmentation de données** (train uniquement) :
- Flip horizontal + vertical
- Rotation ±20°, Zoom ±15%, Translation ±10%
- Variation de contraste ±10%, Luminosité ±10%

---

## Architecture

### Pourquoi ResNet50V2 ?

Après comparaison de 4 architectures :

| Architecture | Params | Val Accuracy | Test Accuracy | Choix |
|---|---|---|---|---|
| CNN Simple (baseline) | ~250K | ~75% | ~73% | — |
| MobileNetV3Large | ~4M | ~92% | ~91% | Trop léger |
| EfficientNetV2S | ~20M | ~95% | ~94% | — |
| **ResNet50V2** | **~25M** | **~98%** | **~98%** | **✅ Gagnant** |

ResNet50V2 a été retenu car il surpasse les autres modèles en **test réel** grâce à :
- **Skip connections** (connexions résiduelles) → gradients stables, pas de dégradation
- **Pré-entraînement ImageNet** → features visuelles riches réutilisables
- **Robustesse** supérieure aux variations d'éclairage, de fond, d'angle

### Entraînement en 2 phases

```
Phase 1 — Feature Extraction (10 époques)
  • Backbone ResNet50V2 gelé (poids ImageNet préservés)
  • Seule la tête de classification est entraînée
  • Optimizer : AdamW (lr=1e-3, weight_decay=1e-4)
  • Callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

Phase 2 — Fine-Tuning (6 époques)
  • 30 dernières couches du backbone dégelées
  • BatchNormalization reste gelé (statistiques ImageNet)
  • Optimizer : AdamW + CosineDecay (lr_initial=1e-5)
```

### Tête de classification

```
ResNet50V2 (backbone)
    ↓ GlobalAveragePooling2D
    ↓ Dense(512) → BatchNorm → ReLU → Dropout(0.4)
    ↓ Dense(8, activation='softmax', dtype=float32)
```

---

## Performances

| Métrique | Valeur |
|----------|--------|
| **Accuracy Top-1** | ~98% |
| **Accuracy Top-3** | ~100% |
| **AUC micro-average** | ~0.997 |
| **Perte test** | ~0.08 |

Les cartes **Grad-CAM** confirment que le modèle se concentre sur les zones de pourriture visibles plutôt que sur l'arrière-plan.

---

## Structure du projet

```
fruit-disease-classifier/
│
├── notebook_brouillon.ipynb      # Toutes les expériences (activations, LR, régularisation, architectures)
├── notebook_final.ipynb           # Pipeline complet et propre (ResNet50V2)
├── presentation_soutenance.pptx  # Slides de présentation (8 slides)
├── README.md                      # Ce fichier
│
├── models/                        # Modèles sauvegardés
│   ├── resnet_best_fe.keras       # Meilleur modèle Phase 1
│   ├── resnet_best_ft.keras       # Meilleur modèle Phase 2 (final)
│   ├── results.json               # Métriques de performance
│   └── resnet50v2_history.json   # Historique d'entraînement
│
└── working/
    └── filtered_dataset/          # Dataset filtré (8 classes)
```

---

## Exécution

### Prérequis

```bash
pip install tensorflow>=2.15 matplotlib seaborn scikit-learn pillow pandas numpy opencv-python
```

Ou sur **Kaggle** (recommandé — GPU T4 gratuit) :
```python
# Les dépendances sont déjà installées
# Monter le dataset : muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
```

### Lancer le notebook final

```bash
# Option 1 : Kaggle (recommandé)
# Importer notebook_final.ipynb dans un nouveau Kaggle notebook
# Ajouter le dataset et lancer "Run All"

# Option 2 : Google Colab
# File > Upload notebook > notebook_final.ipynb
# Activer GPU : Runtime > Change runtime type > T4 GPU

# Option 3 : Local (nécessite GPU NVIDIA)
jupyter notebook notebook_final.ipynb
```


---

## Expériences

Le **notebook de brouillon** documente toutes les expériences menées :

| Expérience | Variable testée | Résultat clé |
|---|---|---|
| 1 — Learning Rate | 1e-2, 1e-3, 5e-4, 1e-4 | LR=1e-3 optimal |
| 2 — Régularisation | Dropout, L2, combinaison | Dropout+L2 réduit le gap train-val |
| 3 — ResNet50V2 | Transfer learning 2 phases | **~98% val acc** |
| 4 — MobileNetV3 | Architecture légère | ~92% val acc |
| 5 — EfficientNetV2S | Architecture haute précision | ~95% val acc |
| 6 — Fine-tuning | Nombre de couches dégelées | 30 couches optimal |

---

## Licence

Projet académique — Dataset sous licence Kaggle (usage non-commercial).
