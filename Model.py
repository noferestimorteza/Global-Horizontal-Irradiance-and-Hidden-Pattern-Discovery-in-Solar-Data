import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from mlxtend.frequent_patterns import fpgrowth, association_rules

output_path = ''
df = pd.read_csv('dataset.csv')
df['period_end'] = pd.to_datetime(df['period_end'])
df['hour'] = df['period_end'].dt.hour
df['month'] = df['period_end'].dt.month

def create_target(row):
    if row['ghi'] <= 5: return 0           # Night
    if row['cloud_opacity'] < 15: return 1 # High Prod
    if row['cloud_opacity'] < 45: return 2 # Medium Prod
    return 3                               # Low Prod

df['target'] = df.apply(create_target, axis=1)

features = ['air_temp', 'cloud_opacity', 'dewpoint_temp', 'relative_humidity', 
            'surface_pressure', 'wind_speed_10m', 'hour', 'month']
X = df[features]
y = df['target']

print("="*60)
print("1. تحلیل پیش‌پردازش (Preprocessing Analysis)")
print("="*60)
print("\n>>> ماتریس کوواریانس ویژگی‌ها:")
print(X.cov())

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='RdYlGn', fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig(output_path + 'correlation_matrix.png')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes (BN)": GaussianNB(),
    "Linear SVM": LinearSVC(max_iter=2000)
}

results_summary = []

print("\n" + "="*60)
print("2. ارزیابی الگوریتم‌های طبقه‌بندی (Classification Results)")
print("="*60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results_summary.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})
    
    print(f"\n📊 خروجی مدل: {name}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(output_path + f'cm_{name.replace(" ", "_")}.png')
    plt.show()

print("\n" + "="*60)
print("3. استخراج الگوهای تکرار شونده (FP-Growth)")
print("="*60)
df_fp = X.copy()
for col in df_fp.columns:
    df_fp[col] = (df_fp[col] > df_fp[col].mean()).astype(bool)

frequent_itemsets = fpgrowth(df_fp, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.to_csv(output_path + 'frequent_patterns.csv')
print(f"\n✅ تعداد {len(rules)} قانون انجمنی استخراج و در فایل CSV ذخیره شد.")

res_df = pd.DataFrame(results_summary)
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Accuracy', data=res_df, palette='viridis')
plt.title("Model Comparison - Accuracy Score")
plt.savefig(output_path + 'model_comparison.png')
plt.show()

print("\n" + "="*60)
print("🏁 پایان پروژه: تمامی فایل‌ها و نمودارها در Google Drive ذخیره شدند.")
print(f"📁 مسیر: {output_path}")
print("="*60)
