"""
Modelo v3: Una reacción canónica por tipo de compuesto.
Los features del compuesto predicen el tipo de reacción más típica.
Esto es coherente: dado el perfil molecular, ¿cuál es la reacción dominante?
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib, json

FEATURE_COLUMNS = [
    'n_elementos','total_atomos','masa_molar','n_H','n_C','n_N','n_O',
    'n_S','n_P','n_halogen','n_metal','n_nonmetal','hc_ratio','ho_ratio',
    'co_ratio','no_ratio','tiene_OH','tiene_COOH','tiene_CO3','tiene_SO4',
    'tiene_NO3','tiene_NO2','tiene_PO4','tiene_NH','es_diatomica',
    'es_homoatomica','es_acido','es_base','es_sal','es_oxido','es_organica',
    'tiene_carbono','tiene_nitrogeno','tiene_oxigeno','tiene_fosforo',
    'tiene_azufre','tiene_metal','n_tipos_metal','es_nitrato','es_nitrito',
    'es_fluoruro','es_cloruro','es_bromuro','es_yoduro','ratio_O_total',
    'ratio_metal_total','es_oxoacido','es_anhidrido','n_enlaces_estimados',
    'tipo_enlace','n_atomos_3d','n_bonds_3d','spread_3d','max_dist_3d'
]

# UNA reacción canónica por tipo de compuesto (la más representativa/educativa)
REACCION_CANONICA = {
    "Óxido":                {"tipo": "neutralización", "patron": "MO + H₂O → M(OH)₂ | MO + HCl → MCl₂ + H₂O", "descripcion": "Óxido + agua → hidróxido; óxido + ácido → sal + agua"},
    "Ácido":                {"tipo": "neutralización", "patron": "HnA + nMOH → MnA + nH₂O", "descripcion": "Ácido + base → sal + agua (reacción de neutralización)"},
    "Base/Hidróxido":       {"tipo": "precipitación", "patron": "M(OH)₂ + MX₂ → M'(OH)₂↓ + MX₂ | MOH + HA → MA + H₂O", "descripcion": "Hidróxido precipita metales; neutraliza ácidos"},
    "Haluro":               {"tipo": "precipitación", "patron": "MX + AgNO₃ → AgX↓ + M(NO₃) | MX + NaOH → M(OH)↓ + NaX", "descripcion": "Haluro + plata → precipitado; con NaOH → hidróxido"},
    "Carbonato":            {"tipo": "descomposición", "patron": "MCO₃ + 2HA → MA₂ + H₂O + CO₂↑ | MCO₃ → MO + CO₂", "descripcion": "Carbonato + ácido → efervescencia CO₂; calcinación"},
    "Sulfato":              {"tipo": "precipitación", "patron": "MSO₄ + BaCl₂ → BaSO₄↓ + MCl₂", "descripcion": "Precipitación de BaSO₄ (prueba cualitativa de sulfatos)"},
    "Nitrato/Nitrito":      {"tipo": "descomposición", "patron": "2MNO₃ → 2MNO₂ + O₂ (calor) | MNO₃ + HA → MA + HNO₃", "descripcion": "Nitrato se descompone con calor; con ácido → ácido nítrico"},
    "Fosfato":              {"tipo": "precipitación", "patron": "M₃PO₄ + CaCl₂ → Ca₃(PO₄)₂↓ | M₃PO₄ + 3HA → MA + H₃PO₄", "descripcion": "Precipitación de fosfato cálcico; neutralización"},
    "Sulfuro":              {"tipo": "desplazamiento", "patron": "MS + 2HA → MA₂ + H₂S↑", "descripcion": "Sulfuro + ácido → H₂S (gas tóxico característico)"},
    "Sulfito":              {"tipo": "desplazamiento", "patron": "MSO₃ + 2HA → MA₂ + H₂O + SO₂↑", "descripcion": "Sulfito + ácido → SO₂ (gas asfixiante)"},
    "Fosfito":              {"tipo": "oxidación", "patron": "M₃PO₃ + H₂O₂ → M₃PO₄ + H₂O", "descripcion": "Fosfito se oxida a fosfato con agentes oxidantes"},
    "Peróxido":             {"tipo": "hidrólisis", "patron": "M₂O₂ + H₂O → 2MOH + ½O₂↑", "descripcion": "Peróxido + agua → base + oxígeno (generación de O₂)"},
    "Cromato/Dicromato":    {"tipo": "oxidación-reducción", "patron": "Cr₂O₇²⁻ + 6e⁻ + 14H⁺ → 2Cr³⁺ + 7H₂O", "descripcion": "Dicromato: oxidante fuerte; cromato⇌dicromato según pH"},
    "Permanganato":         {"tipo": "oxidación-reducción", "patron": "MnO₄⁻ + 5e⁻ + 8H⁺ → Mn²⁺ + 4H₂O (ácido)", "descripcion": "Permanganato: oxidante muy fuerte (titulaciones redox)"},
    "Carburo/Nitruro":      {"tipo": "hidrólisis", "patron": "CaC₂ + 2H₂O → Ca(OH)₂ + C₂H₂↑ | Ca₃N₂ + 6H₂O → 3Ca(OH)₂ + 2NH₃↑", "descripcion": "Carburo/nitruro + agua → gas + hidróxido"},
    "Cianuro":              {"tipo": "desplazamiento", "patron": "MCN + HA → MA + HCN↑ (¡GAS LETAL!)", "descripcion": "Cianuro + ácido → HCN (extremadamente tóxico)"},
    "Hidruro":              {"tipo": "hidrólisis", "patron": "MH + H₂O → MOH + H₂↑", "descripcion": "Hidruro activo + agua → base + hidrógeno gas"},
    "Tiosulfato":           {"tipo": "oxidación-reducción", "patron": "2S₂O₃²⁻ + I₂ → S₄O₆²⁻ + 2I⁻ (yodometría)", "descripcion": "Tiosulfato reduce yodo (titulación yodométrica)"},
    "Silicato":             {"tipo": "precipitación", "patron": "MSiO₃ + 2HA → MA₂ + H₂SiO₃↓ (gel)", "descripcion": "Silicato + ácido → ácido silícico (gel)"},
    "Oxalato":              {"tipo": "oxidación-reducción", "patron": "C₂O₄²⁻ + MnO₄⁻ + H⁺ → CO₂ + Mn²⁺ (permanganometría)", "descripcion": "Oxalato reduce permanganato (valoración)"},
    "Acetato":              {"tipo": "desplazamiento", "patron": "CH₃COOM + HA → MA + CH₃COOH", "descripcion": "Acetato + ácido fuerte → ácido acético (vinagre)"},
    "Borato":               {"tipo": "neutralización", "patron": "Na₂B₄O₇ + 2HCl + 5H₂O → 4H₃BO₃ + 2NaCl", "descripcion": "Bórax + ácido → ácido bórico"},
    "Molécula homoatómica": {"tipo": "síntesis", "patron": "H₂ + ½O₂ → H₂O | N₂ + 3H₂ → 2NH₃ | Cl₂ + 2NaOH → NaOCl", "descripcion": "Moléculas diatómicas: combustión, síntesis, dismutación"},
    "Compuesto inorgánico": {"tipo": "síntesis", "patron": "NH₃ + HCl → NH₄Cl | H₂O + SO₃ → H₂SO₄", "descripcion": "Reacciones de combinación directa"},
    "Alcano":               {"tipo": "combustión", "patron": "CₙH₂ₙ₊₂ + O₂ → CO₂ + H₂O | CₙH₂ₙ₊₂ + X₂ → CₙH₂ₙ₊₁X + HX", "descripcion": "Combustión completa; halogenación radical"},
    "Alqueno/Alquino":      {"tipo": "adición-sustitución", "patron": "C=C + Br₂ → CBr-CBr | C=C + H₂O → C-OH | nC=C → polímero", "descripcion": "Adición electrofílica, hidratación, polimerización"},
    "Alcohol":              {"tipo": "oxidación", "patron": "R-OH [O] → R-CHO → R-COOH | R-OH + Na → RO⁻Na⁺ + H₂↑", "descripcion": "Oxidación a aldehído/ácido; reacción con sodio"},
    "Compuesto carbonílico":{"tipo": "adición-sustitución", "patron": "R-CHO + Cu(OH)₂ → Cu₂O↓ (Fehling) | R-CHO + 2Ag⁺ → 2Ag↓ (espejo)", "descripcion": "Aldehídos: reductores (Fehling, Tollens)"},
    "Aminoácido":           {"tipo": "adición-sustitución", "patron": "nH₂N-CHR-COOH → polipéptido + nH₂O | + ninhidrina → color púrpura", "descripcion": "Policondensación (proteínas); detección con ninhidrina"},
    "Carbohidrato":         {"tipo": "oxidación", "patron": "C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O | C₆H₁₂O₆ → 2C₂H₅OH + 2CO₂", "descripcion": "Respiración celular; fermentación alcohólica"},
    "Amina":                {"tipo": "neutralización", "patron": "R-NH₂ + HA → R-NH₃⁺A⁻ | C₆H₅NH₂ + HNO₂ → C₆H₅N₂⁺ (diazonio)", "descripcion": "Amina básica + ácido; diazotación de arilaminas"},
    "Alcaloide":            {"tipo": "neutralización", "patron": "Alcaloide + HA → sal de alcaloide (soluble) | + reactivos colorimétricos", "descripcion": "Alcaloides: bases → sales solubles con ácidos"},
}

df = pd.read_excel('/mnt/user-data/outputs/dataset_v3.xlsx')
df['tipo_compuesto'] = df['tipo_compuesto'].str.strip()

# Asignar reacción canónica
sin_cobertura = set()
tipos_reaccion, patrones, descripciones = [], [], []
for _, row in df.iterrows():
    tipo = row['tipo_compuesto']
    if tipo in REACCION_CANONICA:
        r = REACCION_CANONICA[tipo]
        tipos_reaccion.append(r['tipo'])
        patrones.append(r['patron'])
        descripciones.append(r['descripcion'])
    else:
        sin_cobertura.add(tipo)
        tipos_reaccion.append('otras reacciones')
        patrones.append('reacción no clasificada')
        descripciones.append('tipo de compuesto sin reacción definida')

df['tipo_reaccion'] = tipos_reaccion
df['patron_reaccion'] = patrones
df['descripcion_reaccion'] = descripciones

if sin_cobertura:
    print(f"Sin cobertura: {sin_cobertura}")

print("Distribución:")
print(df['tipo_reaccion'].value_counts().to_string())
print(f"\nClases: {df['tipo_reaccion'].nunique()}")

# Preparar features
feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
X = df[feature_cols].copy()
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(0)

y = df['tipo_reaccion']
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

acc_test = accuracy_score(y_test, rf.predict(X_test))
acc_train = accuracy_score(y_train, rf.predict(X_train))
cv = cross_val_score(rf, X, y_enc, cv=5)

print(f"\nAccuracy train: {acc_train:.3f}")
print(f"Accuracy test:  {acc_test:.3f}")
print(f"CV 5-fold: {cv.mean():.3f} ± {cv.std():.3f}")

# Exportar JSON
def serialize(tree, node=0):
    if tree.children_left[node] == -1:
        return [int(tree.value[node].argmax())]
    return [int(tree.feature[node]), round(float(tree.threshold[node]),6),
            serialize(tree, tree.children_left[node]),
            serialize(tree, tree.children_right[node])]

# Incluir mapa tipo_compuesto → reacción para lookup directo en browser
tipo_to_reaccion = {k: v for k, v in REACCION_CANONICA.items()}

model_data = {
    "trees": [serialize(e.tree_) for e in rf.estimators_],
    "classes": list(le.classes_),
    "features": feature_cols,
    "n_trees": len(rf.estimators_),
    "tipo_to_reaccion": tipo_to_reaccion
}

with open('/home/claude/reaction_model_v3.json','w') as f:
    json.dump(model_data, f, separators=(',',':'), ensure_ascii=False)

size = len(json.dumps(model_data, separators=(',',':'), ensure_ascii=False).encode())/1024
print(f"\nJSON: reaction_model_v3.json ({size:.1f} KB)")

# Guardar dataset y modelo
df_out = df[['formula','tipo_compuesto','tipo_reaccion','patron_reaccion','descripcion_reaccion']].copy()
df_out.to_excel('/mnt/user-data/outputs/dataset_reacciones_v3.xlsx', index=False)

joblib.dump({'model': rf, 'encoder': le, 'features': feature_cols,
             'tipo_to_reaccion': tipo_to_reaccion},
            '/mnt/user-data/outputs/modelo_reacciones_v3.joblib')
print("Archivos guardados correctamente.")
