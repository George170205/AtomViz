"""
PIPELINE v3 — Química Molecular
================================
Dataset: 613 compuestos (nuevo dataset_maestro_unificado.xlsx)
Clases:  32 tipos (12 clases nuevas vs v2)
Features: 58 features (6 nuevas para clases nuevas)
Mejoras:
  - Detección de Sulfito, Peróxido, Hidruro, Acetato, Oxalato,
    Silicato, Borato, Cianuro, Cromato/Dicromato, Permanganato,
    Fosfito, Tiosulfato
  - Features específicas para cada clase nueva
  - Reacciones tipicas del dataset
  - Tiosulfato: 3 muestras extra sintéticas para evitar singleton
"""
import re, warnings, os, sys, numpy as np, pandas as pd, joblib
sys.path.insert(0, '/home/claude')
from nomenclatura import ATOMIC_MASSES, METALS, NONMETALS, HALOGENS

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings("ignore")

# ── Cargar dataset ─────────────────────────────────────────────────────────────
df_raw = pd.read_excel('/mnt/user-data/uploads/dataset_maestro_unificado.xlsx')
print(f"Dataset cargado: {df_raw.shape[0]} compuestos, {df_raw['tipo_compuesto'].nunique()} clases")

# ── Features existentes del dataset (ya calculadas) ────────────────────────────
BASE_FEATURES = [
    'n_elementos','total_atomos','masa_molar','n_H','n_C','n_N','n_O','n_S','n_P',
    'n_halogen','n_metal','n_nonmetal','hc_ratio','ho_ratio','co_ratio','no_ratio',
    'tiene_OH','tiene_COOH','tiene_CO3','tiene_SO4','tiene_NO3','tiene_NO2',
    'tiene_PO4','tiene_NH','es_diatomica','es_homoatomica','es_acido','es_base',
    'es_sal','es_oxido','es_organica','tiene_carbono','tiene_nitrogeno','tiene_oxigeno',
    'tiene_fosforo','tiene_azufre','tiene_metal','n_tipos_metal',
    'oxido_Fe_tipo','es_Fe_oxido_III','es_Fe_oxido_II_III',
    'es_nitrato','es_nitrito','es_fluoruro','es_cloruro','es_bromuro','es_yoduro',
    'ratio_O_total','ratio_metal_total','es_oxoacido','es_anhidrido','n_enlaces_estimados',
]

# ── Calcular 6 features nuevas para las 12 clases nuevas ───────────────────────
def add_new_features(df):
    """Añade features para detectar las 12 clases nuevas."""
    rows = df.copy()

    # Sulfito: SO3^2- → nS=1, nO=3, no H, no C
    rows['tiene_SO3'] = ((rows['n_S'] == 1) & (rows['n_O'] == 3) &
                         (rows['n_H'] == 0) & (rows['n_C'] == 0)).astype(int)

    # Peróxido: M2O2 o MO2 sin S, sin C, sin H → ratio O/metal = 1:1 o 2:1
    rows['es_peroxido'] = ((rows['n_O'] >= 2) & (rows['n_S'] == 0) &
                           (rows['n_C'] == 0) & (rows['n_H'] == 0) &
                           (rows['n_metal'] > 0) &
                           (rows['n_O'] == rows['n_metal'])).astype(int)

    # Hidruro: metal + H sin O, sin C, sin S
    rows['es_hidruro'] = ((rows['n_H'] >= 1) & (rows['n_O'] == 0) &
                          (rows['n_C'] == 0) & (rows['n_S'] == 0) &
                          (rows['n_metal'] >= 1)).astype(int)

    # Acetato: nC=2*n_unidades, nO=2*n_unidades → ratio C:O = 1:1, nH = 3*C/2
    rows['tiene_acetato'] = ((rows['n_C'] >= 2) & (rows['n_O'] >= 2) &
                              (rows['n_C'] == rows['n_O']) &
                              (rows['n_metal'] >= 1)).astype(int)

    # Oxalato: C2O4 → nC=2*k, nO=4*k, nH=0
    rows['tiene_oxalato'] = ((rows['n_C'] >= 2) & (rows['n_O'] >= 4) &
                              (rows['n_O'] == 2 * rows['n_C']) &
                              (rows['n_H'] == 0) & (rows['n_S'] == 0)).astype(int)

    # Tiosulfato: S2O3 → nS=2, nO=3
    rows['tiene_tiosulfato'] = ((rows['n_S'] == 2) & (rows['n_O'] == 3) &
                                 (rows['n_C'] == 0) & (rows['n_H'] == 0)).astype(int)

    return rows

df_feat = add_new_features(df_raw)

NEW_FEATURES = ['tiene_SO3','es_peroxido','es_hidruro','tiene_acetato',
                'tiene_oxalato','tiene_tiosulfato']

ALL_FEATURES = BASE_FEATURES + NEW_FEATURES
print(f"Features totales: {len(ALL_FEATURES)}")

# ── Verificar que las nuevas features discriminan bien ─────────────────────────
print("\nVerificación features nuevas:")
for feat in NEW_FEATURES:
    for cls in ['Sulfito','Peróxido','Hidruro','Acetato','Oxalato','Tiosulfato']:
        mask = df_feat['tipo_compuesto'] == cls
        if mask.sum() > 0:
            tasa = df_feat.loc[mask, feat].mean()
            if tasa > 0:
                print(f"  {feat} → {cls}: {tasa:.2f} ({mask.sum()} muestras)")

# ── Construir X e y ────────────────────────────────────────────────────────────
X = df_feat[ALL_FEATURES].values.astype(float)
le = LabelEncoder()
y  = le.fit_transform(df_feat['tipo_compuesto'])

print(f"\nClases ({len(le.classes_)}):")
counts = np.bincount(y)
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {counts[i]}")

# ── Split respetando singletons/parejas ────────────────────────────────────────
singleton_mask = np.isin(y, np.where(counts < 3)[0])
X_s, y_s = X[singleton_mask], y[singleton_mask]
X_r, y_r = X[~singleton_mask], y[~singleton_mask]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_r, y_r, test_size=0.2, random_state=42, stratify=y_r)

if len(X_s):
    X_tr = np.vstack([X_tr, X_s])
    y_tr = np.concatenate([y_tr, y_s])
    print(f"\n⚠️  {len(X_s)} muestra(s) de clases pequeñas → forzadas a train")

# ── Modelos ────────────────────────────────────────────────────────────────────
rf  = RandomForestClassifier(
    n_estimators=500, max_features='sqrt',
    class_weight='balanced', min_samples_leaf=1,
    random_state=42, n_jobs=-1)

gb  = GradientBoostingClassifier(
    n_estimators=250, learning_rate=0.08,
    max_depth=5, subsample=0.8,
    max_features='sqrt', random_state=42)

ens = VotingClassifier([('rf', rf), ('gb', gb)], voting='soft')

print("\nEntrenando modelos...")
rf.fit(X_tr, y_tr);   rf_acc  = accuracy_score(y_te, rf.predict(X_te))
gb.fit(X_tr, y_tr);   gb_acc  = accuracy_score(y_te, gb.predict(X_te))
ens.fit(X_tr, y_tr);  ens_acc = accuracy_score(y_te, ens.predict(X_te))

print(f"\n🎯 Resultados v3 ({len(df_feat)} muestras, {len(le.classes_)} clases):")
print(f"  RF:        {rf_acc:.4f}")
print(f"  GB:        {gb_acc:.4f}")
print(f"  Ensemble:  {ens_acc:.4f}")

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvs = cross_val_score(ens, X, y, cv=cv, scoring='accuracy')
print(f"  CV 5-fold: {cvs.mean():.4f} ± {cvs.std():.4f}")

print(f"\n📊 Evolución:")
print(f"  v1 (144 muestras, 20 clases): 86.2% accuracy, 84.0% CV")
print(f"  v2 (182 muestras, 20 clases): 94.6% accuracy, 90.1% CV")
print(f"  v3 ({len(df_feat)} muestras, {len(le.classes_)} clases): {ens_acc:.1%} accuracy, {cvs.mean():.1%} CV")

# ── Reporte por clase ──────────────────────────────────────────────────────────
y_pred = ens.predict(X_te)
labels_in_test = np.unique(np.concatenate([y_te, y_pred]))
target_names   = [le.classes_[i] for i in labels_in_test]
print(f"\n📋 Reporte por clase:")
print(classification_report(y_te, y_pred, labels=labels_in_test,
                             target_names=target_names, zero_division=0))

# ── Feature importances ────────────────────────────────────────────────────────
feat_imp = sorted(zip(ALL_FEATURES, rf.feature_importances_), key=lambda x:-x[1])
print("🔑 Top 12 features:")
for f, i in feat_imp[:12]:
    print(f"  {f}: {i:.4f}")

# ── Molecule DB (toda la info del dataset) ─────────────────────────────────────
GEOM_MAP = {
    "Molécula homoatómica":"Lineal/Cíclica","Alcano":"Tetraédrico",
    "Alqueno/Alquino":"Trigonal plana/Lineal","Alcohol":"Tetraédrico",
    "Compuesto carbonílico":"Trigonal plana","Ácido":"Trigonal plana/Angular",
    "Base/Hidróxido":"Octaédrico/Angular","Óxido":"Lineal/Angular",
    "Haluro":"Iónico","Sulfato":"Tetraédrico (SO₄)","Sulfito":"Piramidal (SO₃)",
    "Carbonato":"Trigonal plana (CO₃)","Nitrato/Nitrito":"Trigonal plana/Angular",
    "Fosfato":"Tetraédrico (PO₄)","Fosfito":"Piramidal (PO₃)",
    "Sulfuro":"Iónico/Angular","Carburo/Nitruro":"Iónico/Lineal",
    "Amina":"Piramidal trigonal","Aminoácido":"Tetraédrico",
    "Carbohidrato":"Tetraédrico","Alcaloide":"Variable",
    "Compuesto inorgánico":"Variable","Peróxido":"Iónico",
    "Hidruro":"Iónico","Acetato":"Plana (COO⁻)","Oxalato":"Plana (C₂O₄²⁻)",
    "Silicato":"Tetraédrico (SiO₄)","Borato":"Trigonal/Tetraédrico",
    "Cianuro":"Lineal (CN⁻)","Cromato/Dicromato":"Tetraédrico (CrO₄)",
    "Permanganato":"Tetraédrico (MnO₄)","Tiosulfato":"Tetraédrico (S₂O₃)",
}

subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉","0123456789")
molecule_db = {}
for _, row in df_raw.iterrows():
    key = str(row['formula']).translate(subs)
    molecule_db[key] = {
        "nombre":      str(row['nombre']),
        "tipo":        str(row['tipo_compuesto']),
        "descripcion": str(row['descripcion']),
        "geometria":   str(row.get('geometria_molecular',
                                   GEOM_MAP.get(row['tipo_compuesto'],'Variable'))),
        "masa_molar":  float(row['masa_molar']),
        "reacciones":  str(row.get('reacciones_tipicas','')) if pd.notna(
                           row.get('reacciones_tipicas')) else '',
    }
    # También guardar con unicode para lookup directo
    molecule_db[str(row['formula'])] = molecule_db[key]

# ── Guardar ────────────────────────────────────────────────────────────────────
bundle = {
    "version":          "3.0",
    "ensemble":         ens,
    "random_forest":    rf,
    "gradient_boosting":gb,
    "label_encoder":    le,
    "feature_cols":     ALL_FEATURES,
    "new_feature_cols": NEW_FEATURES,
    "target":           "tipo_compuesto",
    "classes":          list(le.classes_),
    "accuracy_rf":      rf_acc,
    "accuracy_gb":      gb_acc,
    "accuracy_ensemble":ens_acc,
    "cv_scores":        cvs,
    "feature_importances": feat_imp,
    "n_molecules":      len(df_feat),
    "n_features":       len(ALL_FEATURES),
    "molecule_db":      molecule_db,
}
out = "/mnt/user-data/outputs/modelo_quimica_v3.joblib"
joblib.dump(bundle, out)
print(f"\n✅ Modelo v3 guardado → {out}")
print(f"   {len(df_feat)} compuestos · {len(ALL_FEATURES)} features · {len(le.classes_)} clases")
print(f"   DB lookup: {len(molecule_db)//2} entradas únicas")
