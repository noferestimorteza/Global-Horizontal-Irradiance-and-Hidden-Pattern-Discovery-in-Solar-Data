import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, f1_score, roc_curve, auc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})
PALETTE = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
CLASS_NAMES = ['Night (0)', 'High Prod (1)', 'Med Prod (2)', 'Low Prod (3)']

output_path = ''

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('dataset.csv')
df['period_end'] = pd.to_datetime(df['period_end'])
df['hour']  = df['period_end'].dt.hour
df['month'] = df['period_end'].dt.month

def create_target(row):
    if row['ghi'] <= 5:          return 0
    if row['cloud_opacity'] < 15: return 1
    if row['cloud_opacity'] < 45: return 2
    return 3

df['target'] = df.apply(create_target, axis=1)

features = ['air_temp', 'cloud_opacity', 'dewpoint_temp', 'relative_humidity',
            'surface_pressure', 'wind_speed_10m', 'hour', 'month']
X = df[features]
y = df['target']

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("1. Preprocessing Analysis")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(X.corr(), dtype=bool))
sns.heatmap(X.corr(), annot=True, cmap='RdYlGn', fmt=".2f",
            mask=mask, ax=ax, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title("Figure 1 – Feature Correlation Matrix", fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(output_path + 'fig1_correlation_matrix.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Class Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = y.value_counts().sort_index()
bars = axes[0].bar(CLASS_NAMES, counts.values, color=PALETTE, edgecolor='white', linewidth=0.8)
axes[0].set_title("Figure 2a – Class Distribution", fontweight='bold')
axes[0].set_ylabel("Sample Count")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:,}', ha='center', va='bottom', fontsize=9)

wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=CLASS_NAMES, colors=PALETTE,
    autopct='%1.1f%%', startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.2})
for t in autotexts: t.set_fontsize(9)
axes[1].set_title("Figure 2b – Class Proportions", fontweight='bold')

plt.tight_layout()
plt.savefig(output_path + 'fig2_class_distribution.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Feature Distributions by Class
# ══════════════════════════════════════════════════════════════════════════════
key_features = ['cloud_opacity', 'air_temp', 'relative_humidity', 'wind_speed_10m']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        data = df[df['target'] == cls_idx][feat].dropna()
        axes[i].hist(data, bins=30, alpha=0.55, color=PALETTE[cls_idx],
                     label=cls_name, density=True)
    axes[i].set_title(f'{feat.replace("_", " ").title()}', fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=7, ncol=2)

fig.suptitle("Figure 3 – Feature Distributions per Production Class",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(output_path + 'fig3_feature_distributions.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# Train / Test Split & Models
# ══════════════════════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Decision Tree":    DecisionTreeClassifier(max_depth=12),
    "KNN (K=5)":        KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (BN)": GaussianNB(),
    "Linear SVM":       LinearSVC(max_iter=2000),
}

results_summary = []
trained_models  = {}

print("\n" + "="*60)
print("2. Classification Results")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    trained_models[name] = (model, y_pred)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    results_summary.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})

    print(f"\n Model: {name}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Confusion Matrices (2×2 grid)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, (model, y_pred)) in enumerate(trained_models.items()):
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=axes[i],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, cbar_kws={'label': '% of True Class'})
    axes[i].set_title(f'{name}', fontweight='bold')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

fig.suptitle("Figure 4 – Normalised Confusion Matrices (%)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(output_path + 'fig4_confusion_matrices.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Model Comparison (Accuracy + F1)
# ══════════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results_summary)
x      = np.arange(len(res_df))
width  = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - width/2, res_df['Accuracy'], width, label='Accuracy',
            color='#2196F3', edgecolor='white')
b2 = ax.bar(x + width/2, res_df['F1-Score'], width, label='Weighted F1',
            color='#4CAF50', edgecolor='white')

for bar in [*b1, *b2]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(res_df['Model'], fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title("Figure 5 – Classifier Comparison: Accuracy vs. Weighted F1-Score",
             fontsize=13, fontweight='bold')
ax.legend()
ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

plt.tight_layout()
plt.savefig(output_path + 'fig5_model_comparison.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Feature Importance (Decision Tree)
# ══════════════════════════════════════════════════════════════════════════════
dt_model = trained_models["Decision Tree"][0]
importances = pd.Series(dt_model.feature_importances_, index=features).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
bars = ax.barh(importances.index, importances.values, color=colors, edgecolor='white')
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_xlabel('Gini Importance')
ax.set_title("Figure 6 – Decision Tree Feature Importance",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'fig6_feature_importance.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Learning Curves (Decision Tree)
# ══════════════════════════════════════════════════════════════════════════════
train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(max_depth=12), X_scaled, y,
    cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='#2196F3', label='Training Accuracy')
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.15, color='#2196F3')
ax.plot(train_sizes, val_scores.mean(axis=1),  's-', color='#E91E63', label='Validation Accuracy')
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.15, color='#E91E63')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Accuracy')
ax.set_title("Figure 7 – Learning Curves (Decision Tree, 5-Fold CV)",
             fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(output_path + 'fig7_learning_curves.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – FP-Growth: Support vs Confidence scatter
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. FP-Growth Association Rules")
print("="*60)

df_fp = X.copy()
for col in df_fp.columns:
    df_fp[col] = (df_fp[col] > df_fp[col].mean()).astype(bool)

frequent_itemsets = fpgrowth(df_fp, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.to_csv(output_path + 'frequent_patterns.csv', index=False)
print(f"\n✅ {len(rules)} association rules extracted and saved to CSV.")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(rules['support'], rules['confidence'],
                c=rules['lift'], cmap='YlOrRd', s=50, alpha=0.7, edgecolors='gray', linewidths=0.3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Lift', fontsize=10)
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_title("Figure 8 – FP-Growth Rules: Support vs. Confidence (color = Lift)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'fig8_fp_growth_rules.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Top-15 Association Rules by Lift
# ══════════════════════════════════════════════════════════════════════════════
top_rules = rules.nlargest(15, 'lift').copy()
top_rules['rule'] = (top_rules['antecedents'].apply(lambda x: ', '.join(list(x))) +
                     '  →  ' +
                     top_rules['consequents'].apply(lambda x: ', '.join(list(x))))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(top_rules)), top_rules['lift'].values,
               color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_rules))), edgecolor='white')
ax.set_yticks(range(len(top_rules)))
ax.set_yticklabels(top_rules['rule'].values, fontsize=8)
ax.set_xlabel('Lift')
ax.set_title("Figure 9 – Top-15 Association Rules by Lift",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'fig9_top_rules.png', bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("All figures saved. Summary:")
figs = [
    'fig1_correlation_matrix.png',
    'fig2_class_distribution.png',
    'fig3_feature_distributions.png',
    'fig4_confusion_matrices.png',
    'fig5_model_comparison.png',
    'fig6_feature_importance.png',
    'fig7_learning_curves.png',
    'fig8_fp_growth_rules.png',
    'fig9_top_rules.png',
]
for f in figs:
    print(f"  ✅ {f}")
print("="*60)