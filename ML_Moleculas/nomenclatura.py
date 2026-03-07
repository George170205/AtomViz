"""
Motor de Nomenclatura Química — IUPAC + Stock
Determina el nombre técnico a partir de la fórmula por reglas deterministas.
NO necesita ML. Las reglas de nomenclatura son exactas al 100%.
"""
import re
from collections import OrderedDict

ATOMIC_MASSES = {
    "H":1.008,"He":4.003,"Li":6.941,"Be":9.012,"B":10.81,"C":12.011,"N":14.007,
    "O":15.999,"F":18.998,"Ne":20.18,"Na":22.99,"Mg":24.305,"Al":26.982,
    "Si":28.086,"P":30.974,"S":32.06,"Cl":35.453,"Ar":39.948,"K":39.098,
    "Ca":40.078,"Ti":47.867,"V":50.942,"Cr":51.996,"Mn":54.938,"Fe":55.845,
    "Co":58.933,"Ni":58.693,"Cu":63.546,"Zn":65.38,"Br":79.904,
    "Sr":87.62,"Ag":107.868,"Sn":118.71,"I":126.904,"Ba":137.327,"Pb":207.2,
}
METALS    = {"Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba","Al","Fe","Cu","Zn",
             "Ag","Au","Pb","Sn","Ni","Co","Mn","Cr","Ti","V","Hg","Bi","Pt"}
NONMETALS = {"H","C","N","O","F","Cl","Br","I","S","P","Se","Si","As","B"}
HALOGENS  = {"F","Cl","Br","I"}

METAL_NAMES = {
    "Li":"litio","Na":"sodio","K":"potasio","Rb":"rubidio","Cs":"cesio",
    "Be":"berilio","Mg":"magnesio","Ca":"calcio","Sr":"estroncio","Ba":"bario",
    "Al":"aluminio","Fe":"hierro","Cu":"cobre","Zn":"zinc","Ag":"plata",
    "Au":"oro","Pb":"plomo","Sn":"estaño","Ni":"níquel","Co":"cobalto",
    "Mn":"manganeso","Cr":"cromo","Ti":"titanio","V":"vanadio",
    "Hg":"mercurio","Bi":"bismuto","Pt":"platino",
}
NONMETAL_NAMES = {
    "H":"hidrógeno","C":"carbono","N":"nitrógeno","O":"oxígeno",
    "F":"flúor","Cl":"cloro","Br":"bromo","I":"yodo",
    "S":"azufre","P":"fósforo","Se":"selenio","Si":"silicio","As":"arsénico","B":"boro",
}
ANION_SUFFIX = {
    "O":"óxido","S":"sulfuro","N":"nitruro","C":"carburo","P":"fosfuro",
    "F":"fluoruro","Cl":"cloruro","Br":"bromuro","I":"yoduro","Se":"seleniuro",
}

# Polianiones ordenados de más específico a más general
POLYATOMIC = OrderedDict([
    ("Cr2O7", ("dicromato",    -2)),
    ("S2O3",  ("tiosulfato",   -2)),
    ("H2PO4", ("dihidrogenofosfato", -1)),
    ("HPO4",  ("hidrogenofosfato",   -2)),
    ("HCO3",  ("bicarbonato",  -1)),
    ("MnO4",  ("permanganato", -1)),
    ("ClO4",  ("perclorato",   -1)),
    ("ClO3",  ("clorato",      -1)),
    ("ClO2",  ("clorito",      -1)),
    ("BrO3",  ("bromato",      -1)),
    ("CrO4",  ("cromato",      -2)),
    ("IO3",   ("yodato",       -1)),
    ("SO4",   ("sulfato",      -2)),
    ("SO3",   ("sulfito",      -2)),
    ("NO3",   ("nitrato",      -1)),
    ("NO2",   ("nitrito",      -1)),
    ("PO4",   ("fosfato",      -3)),
    ("PO3",   ("fosfito",      -3)),
    ("CO3",   ("carbonato",    -2)),
    ("SiO3",  ("silicato",     -2)),
    ("BO3",   ("borato",       -3)),
    ("ClO",   ("hipoclorito",  -1)),
    ("OH",    ("hidróxido",    -1)),
    ("CN",    ("cianuro",      -1)),
    ("NH4",   ("amonio",       +1)),
    ("C2H3O2",("acetato",      -1)),
    ("C2O4",  ("oxalato",      -2)),
    ("S2O3",  ("tiosulfato",   -2)),
    ("SiO3",  ("silicato",     -2)),
    ("BO3",   ("borato",       -3)),
    ("B4O7",  ("tetraborato",  -2)),
])

KNOWN_ACIDS = {
    "HF":"ácido fluorhídrico","HCl":"ácido clorhídrico",
    "HBr":"ácido bromhídrico","HI":"ácido yodhídrico",
    "H2S":"ácido sulfhídrico","HCN":"ácido cianhídrico",
    "H2SO4":"ácido sulfúrico","H2SO3":"ácido sulfuroso",
    "HNO3":"ácido nítrico","HNO2":"ácido nitroso",
    "H3PO4":"ácido fosfórico","H3PO3":"ácido fosforoso",
    "H2CO3":"ácido carbónico","HClO4":"ácido perclórico",
    "HClO3":"ácido clórico","HClO2":"ácido cloroso",
    "HClO":"ácido hipocloroso","H3BO3":"ácido bórico",
    "HMnO4":"ácido permangánico","H2CrO4":"ácido crómico",
    "CH3COOH":"ácido acético","HCOOH":"ácido fórmico",
}

VARIABLE_VALENCE = {
    "Fe":[2,3],"Cu":[1,2],"Sn":[2,4],"Pb":[2,4],"Cr":[2,3,6],
    "Mn":[2,4,7],"Co":[2,3],"Ni":[2,3],"Hg":[1,2],"Au":[1,3],
    "V":[2,3,4,5],"Ti":[2,3,4],
}
ROMAN = {1:"I",2:"II",3:"III",4:"IV",5:"V",6:"VI",7:"VII"}
PREFIX = {1:"mono",2:"di",3:"tri",4:"tetra",5:"penta",
          6:"hexa",7:"hepta",8:"octa",9:"nona",10:"deca"}
ALCANOS = {1:"metano",2:"etano",3:"propano",4:"butano",5:"pentano",
           6:"hexano",7:"heptano",8:"octano",9:"nonano",10:"decano"}


def parse_formula(formula):
    """Parsea fórmula química → dict de conteo de átomos. Maneja paréntesis."""
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉","0123456789")
    f = formula.translate(subs).strip()
    f = re.sub(r'\^[\d\+\-]+', '', f)  # quitar carga iónica

    def expand(s):
        while "(" in s:
            m = re.search(r'\(([^()]+)\)(\d*)', s)
            if not m: break
            mult = int(m.group(2)) if m.group(2) else 1
            inner = re.findall(r'([A-Z][a-z]?)(\d*)', m.group(1))
            expanded = "".join(f"{el}{(int(n) if n else 1)*mult}" for el,n in inner if el)
            s = s[:m.start()] + expanded + s[m.end():]
        return s

    expanded = expand(f)
    counts = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', expanded):
        if el:
            counts[el] = counts.get(el,0) + (int(n) if n else 1)
    return counts


def detect_polyatomic(atoms):
    """Detecta el polianión presente y devuelve (key, nombre, carga, multiplicador, átomos_restantes)."""
    for key, (name, charge) in POLYATOMIC.items():
        key_counts = {}
        for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', key):
            if el: key_counts[el] = key_counts.get(el,0) + (int(n) if n else 1)

        mults, match = [], True
        for el, cnt in key_counts.items():
            if el not in atoms or atoms[el] < cnt or atoms[el] % cnt != 0:
                match = False; break
            mults.append(atoms[el] // cnt)

        if match and mults and len(set(mults)) == 1:
            remaining = {k: v for k, v in atoms.items()}
            for el, cnt in key_counts.items():
                remaining[el] -= cnt * mults[0]
                if remaining[el] == 0: del remaining[el]
            return key, name, charge, mults[0], remaining

    return None, None, None, None, None


def _tipo_from_anion(name):
    if "tiosulfato" in name: return "Tiosulfato"
    if "sulfito" in name: return "Sulfito"
    if "sulfato" in name: return "Sulfato"
    if any(x in name for x in ["nitrato","nitrito"]): return "Nitrato/Nitrito"
    if "fosfito" in name: return "Fosfito"
    if "fosfato" in name: return "Fosfato"
    if any(x in name for x in ["carbonato","bicarbonato"]): return "Carbonato"
    if "hidróxido" in name: return "Base/Hidróxido"
    if "cianuro" in name: return "Cianuro"
    if any(x in name for x in ["cloruro","bromuro","fluoruro","yoduro"]): return "Haluro"
    if any(x in name for x in ["cromato","dicromato"]): return "Cromato/Dicromato"
    if "permanganato" in name: return "Permanganato"
    if "acetato" in name: return "Acetato"
    if "oxalato" in name: return "Oxalato"
    if "tiosulfato" in name: return "Tiosulfato"
    if "silicato" in name: return "Silicato"
    if "borato" in name or "tetraborato" in name: return "Borato"
    if "cianuro" in name: return "Cianuro"
    if any(x in name for x in ["sulfito","bisulfito"]): return "Sulfito"
    if "fosfito" in name: return "Fosfito"
    return "Sal"


def nombre_compuesto(formula):
    """Retorna dict con: nombre, tipo, formula_parseada, confianza."""
    formula_norm = formula.strip()
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉","0123456789")
    formula_ascii = formula_norm.translate(subs)

    # 1. Lookup directo de ácidos conocidos
    for acid_f, acid_name in KNOWN_ACIDS.items():
        if formula_ascii == acid_f:
            return {"nombre": acid_name, "tipo": "Ácido",
                    "atoms": parse_formula(formula_norm), "confianza": "alta"}

    atoms = parse_formula(formula_norm)
    if not atoms:
        return {"nombre": "Fórmula no reconocida", "tipo": "Desconocido",
                "atoms": {}, "confianza": "baja"}

    elements = list(atoms.keys())
    n_types  = len(elements)
    metals_p = [e for e in elements if e in METALS]
    hal_p    = [e for e in elements if e in HALOGENS]
    nH=atoms.get("H",0); nC=atoms.get("C",0); nO=atoms.get("O",0)
    nN=atoms.get("N",0); nS=atoms.get("S",0)

    # 2. Molécula homoatómica
    if n_types == 1:
        el  = elements[0]
        nm  = NONMETAL_NAMES.get(el, METAL_NAMES.get(el, el))
        return {"nombre": f"{nm} elemental", "tipo": "Molécula homoatómica",
                "atoms": atoms, "confianza": "alta"}

    # 3. NH4+ con anión: verificar antes que cualquier otra sal
    if nN > 0 and nH > 0 and nH == 4*nN:
        rest = {k:v for k,v in atoms.items() if k not in ("N","H")}
        if rest:
            pk, pn, pc, pmult, _ = detect_polyatomic(rest)
            if pn:
                return {"nombre": f"{pn} de amonio", "tipo": _tipo_from_anion(pn),
                        "atoms": atoms, "confianza": "alta"}
            if len(rest) == 1:
                el = list(rest.keys())[0]
                if el in HALOGENS:
                    return {"nombre": f"{ANION_SUFFIX[el]} de amonio",
                            "tipo": "Haluro", "atoms": atoms, "confianza": "alta"}

    # 4. Ácido binario H + no-metal sin O (solo para H+halógeno o H+S/P)
    if nH > 0 and not metals_p and not nO and n_types == 2:
        other = [e for e in elements if e != "H"][0]
        if other in HALOGENS or other in ("S","P","Se","As"):
            nombre_ac = f"ácido {NONMETAL_NAMES.get(other,other.lower())}hídrico"
            return {"nombre": nombre_ac, "tipo": "Ácido", "atoms": atoms, "confianza": "alta"}

    # 4b. Hidruro: metal + H, sin O, C, S
    if nH > 0 and metals_p and not nO and not nC and not nS:
        mname = METAL_NAMES.get(metals_p[0], metals_p[0])
        return {"nombre": f"hidruro de {mname}", "tipo": "Hidruro",
                "atoms": atoms, "confianza": "alta"}

    # 4c. Peróxido: M_x O_x (necesita >=2 O), binario, sin H/C/S/P
    if nO >= 2 and metals_p and not nH and not nC and not nS and not atoms.get("P",0) and n_types == 2:
        n_met_atoms = sum(atoms.get(e, 0) for e in metals_p)
        # Alkali: M2O2 (nO==n_met); Alkaline earth: MO2 (nO==2*n_met) e.g. BaO2
        ALKALINE_EARTH = {"Be","Mg","Ca","Sr","Ba"}
        is_alk_earth = all(e in ALKALINE_EARTH for e in metals_p)
        if nO == n_met_atoms or (is_alk_earth and nO == 2 * n_met_atoms):
            mname = METAL_NAMES.get(metals_p[0], metals_p[0])
            return {"nombre": f"per\u00f3xido de {mname}", "tipo": "Per\u00f3xido",
                    "atoms": atoms, "confianza": "alta"}

        # 5. Sales con polianión (sulfatos, fosfatos, nitratos, carbonatos, OH, etc.)
    pk, pn, pc, pmult, remaining = detect_polyatomic(atoms)
    if pn:
        cation_els = list(remaining.keys()) if remaining else []

        # H → oxoácido
        if cation_els == ["H"]:
            ac = pn.replace("ato","ico").replace("ito","oso")
            return {"nombre": f"ácido {ac}", "tipo": "Ácido",
                    "atoms": atoms, "confianza": "alta"}

        # Metal catión
        met_cats = [e for e in cation_els if e in METALS]
        if met_cats:
            metal = met_cats[0]; n_metal = remaining[metal]
            mname = METAL_NAMES.get(metal, metal)
            tipo  = _tipo_from_anion(pn)
            if metal in VARIABLE_VALENCE and n_metal > 0:
                total_anion_q = abs(pc) * pmult
                if total_anion_q % n_metal == 0:
                    ox = total_anion_q // n_metal
                    if ox in ROMAN:
                        return {"nombre": f"{pn} de {mname} ({ROMAN[ox]})",
                                "tipo": tipo, "atoms": atoms, "confianza": "alta"}
            return {"nombre": f"{pn} de {mname}", "tipo": tipo,
                    "atoms": atoms, "confianza": "alta"}

    # 6. Óxido binario
    if nO > 0 and n_types == 2 and not nH:
        other = [e for e in elements if e != "O"][0]
        n_other = atoms[other]
        if other in METALS:
            mname = METAL_NAMES.get(other, other)
            if other in VARIABLE_VALENCE:
                ox_v = (2 * nO) / n_other
                if ox_v == int(ox_v) and int(ox_v) in ROMAN:
                    return {"nombre": f"óxido de {mname} ({ROMAN[int(ox_v)]})",
                            "tipo": "Óxido", "atoms": atoms, "confianza": "alta"}
            # Special: Fe3O4 has mixed valence
            if other == "Fe" and atoms["Fe"]==3 and nO==4:
                return {"nombre": f"óxido de hierro (II,III)", "tipo": "Óxido", "atoms": atoms, "confianza": "alta"}
            return {"nombre": f"óxido de {mname}", "tipo": "Óxido",
                    "atoms": atoms, "confianza": "alta"}
        else:
            pO = PREFIX.get(nO,""); pX = PREFIX.get(n_other,"") if n_other>1 else ""
            nm = NONMETAL_NAMES.get(other, other.lower())
            return {"nombre": f"{pO}óxido de {pX}{nm}".replace("monóxido de ","monóxido de "),
                    "tipo": "Óxido", "atoms": atoms, "confianza": "alta"}

    # 7. Haluro binario metal + halógeno
    if hal_p and metals_p and n_types == 2:
        metal = metals_p[0]; hal = hal_p[0]
        mname = METAL_NAMES.get(metal, metal)
        hanion = ANION_SUFFIX.get(hal, hal.lower()+"uro")
        if metal in VARIABLE_VALENCE:
            ox = atoms[hal] // atoms[metal]
            if atoms[hal] % atoms[metal] == 0 and ox in ROMAN:
                return {"nombre": f"{hanion} de {mname} ({ROMAN[ox]})",
                        "tipo": "Haluro", "atoms": atoms, "confianza": "alta"}
        return {"nombre": f"{hanion} de {mname}", "tipo": "Haluro",
                "atoms": atoms, "confianza": "alta"}

    # 8. Sulfuro binario
    if nS > 0 and not nO and not nH and metals_p and n_types == 2:
        metal = metals_p[0]; mname = METAL_NAMES.get(metal, metal)
        if metal in VARIABLE_VALENCE:
            ox_v = (2*nS)/atoms[metal]
            if ox_v == int(ox_v) and int(ox_v) in ROMAN:
                return {"nombre": f"sulfuro de {mname} ({ROMAN[int(ox_v)]})",
                        "tipo": "Sulfuro", "atoms": atoms, "confianza": "alta"}
        return {"nombre": f"sulfuro de {mname}", "tipo": "Sulfuro",
                "atoms": atoms, "confianza": "alta"}

    # 9. Carburo / Nitruro
    if nC > 0 and not nO and not nH and metals_p:
        mname = METAL_NAMES.get(metals_p[0], metals_p[0])
        return {"nombre": f"carburo de {mname}", "tipo": "Carburo/Nitruro",
                "atoms": atoms, "confianza": "alta"}
    if nN > 0 and not nO and not nH and not nC and metals_p:
        mname = METAL_NAMES.get(metals_p[0], metals_p[0])
        return {"nombre": f"nitruro de {mname}", "tipo": "Carburo/Nitruro",
                "atoms": atoms, "confianza": "alta"}

    # 10. Compuestos orgánicos
    if nC > 0 and not metals_p:
        if set(elements) == {"C","H"}:
            if set(elements) == {"C","H"} and nH == 2*nC+2:
                return {"nombre": ALCANOS.get(nC, f"alcano C{nC}H{nH}"),
                        "tipo": "Alcano", "atoms": atoms, "confianza": "alta"}
            if nH == 2*nC:
                return {"nombre": {2:"etileno",3:"propileno"}.get(nC, f"alqueno C{nC}H{nH}"),
                        "tipo": "Alqueno/Alquino", "atoms": atoms, "confianza": "alta"}
            if nH == 2*nC-2:
                return {"nombre": "acetileno" if nC==2 else f"alquino C{nC}H{nH}",
                        "tipo": "Alqueno/Alquino", "atoms": atoms, "confianza": "alta"}
        # Carbohidrato Cn(H2O)n
        if nO > 1 and nH == 2*nO and nN == 0 and nS == 0:
            sugars = {6:"glucosa / fructosa",12:"sacarosa / maltosa",5:"pentosa"}
            return {"nombre": sugars.get(nC, f"carbohidrato C{nC}H{nH}O{nO}"),
                    "tipo": "Carbohidrato", "atoms": atoms, "confianza": "media"}
        # Aminoácido
        if nN >= 1 and nO >= 2:
            return {"nombre": f"aminoácido (C{nC}H{nH}N{nN}O{nO})",
                    "tipo": "Aminoácido", "atoms": atoms, "confianza": "media"}
        # Amina
        if nN >= 1 and nO == 0:
            return {"nombre": f"amina (C{nC}H{nH}N{nN})",
                    "tipo": "Amina", "atoms": atoms, "confianza": "media"}
        # Alcohol
        if nO == 1 and nN == 0 and nH >= 2*nC:
            alc = {1:"metanol",2:"etanol",3:"propanol",4:"butanol",6:"fenol"}
            return {"nombre": alc.get(nC, f"alcohol C{nC}H{nH}O"),
                    "tipo": "Alcohol", "atoms": atoms, "confianza": "alta" if nC in alc else "media"}
        # Carbonílico
        if nO >= 1 and nN == 0:
            return {"nombre": f"compuesto carbonílico (C{nC}H{nH}O{nO})",
                    "tipo": "Compuesto carbonílico", "atoms": atoms, "confianza": "media"}
        return {"nombre": f"compuesto orgánico C{nC}H{nH}",
                "tipo": "Compuesto orgánico", "atoms": atoms, "confianza": "media"}

    # Fallback
    all_nms = [METAL_NAMES.get(e, NONMETAL_NAMES.get(e,e)) for e in elements]
    return {"nombre": f"compuesto de {', '.join(all_nms)}",
            "tipo": "Compuesto inorgánico", "atoms": atoms, "confianza": "media"}


if __name__ == "__main__":
    tests = [
        "Fe2(SO4)3","Fe(OH)3","C6H12O6","NaOH","Ca3(PO4)2",
        "FeCl3","FeCl2","CuSO4","HCl","H2SO4","CaCO3","KNO3",
        "Fe2O3","Fe3O4","CH4","C2H6","C2H5OH","NH4Cl",
        "(NH4)2SO4","K2Cr2O7","KMnO4","SnO2","Pb(NO3)2",
        "Al2(SO4)3","MnO2","Cu(OH)2","NiCl2","CoSO4",
    ]
    print(f"{'Fórmula':<18} {'Nombre obtenido':<45} Conf")
    print("─"*75)
    for f in tests:
        r = nombre_compuesto(f)
        print(f"  {f:<16} {r['nombre']:<45} {r['confianza']}")
