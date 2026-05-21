import pickle
import os
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
import chembl_db
import chem_builder
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from rdkit.Chem import Descriptors, MACCSkeys
import pandas as pd
from rdkit import Chem
import logging
import warnings
import os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from rdkit.Chem import AllChem
from rdkit import DataStructs
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from rdkit.Chem.MolStandardize import rdMolStandardize

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
os.environ["PYTORCH_LIGHTNING_SUPPRESS_WARNINGS"] = "1"

DB_FILE = 'drugs_db.pkl'
GLOBAL_MOLS = []

if os.path.exists(DB_FILE):
    print(f"📦 [Backend] Loading Pickled database from {DB_FILE}...")
    with open(DB_FILE, 'rb') as f:
        GLOBAL_MOLS = pickle.load(f)
    print(f"✅ [Backend] Successfully loaded {len(GLOBAL_MOLS)} molecules into RAM.")
else:
    print(f"⚠️ [Backend] CRITICAL WARNING: Database file {DB_FILE} not found!")

def fix_chemical_input(text):
    replacements = {
        'с': 'c', 'С': 'C',
        'о': 'o', 'О': 'O',
        'н': 'n', 'Н': 'N',
        'р': 'p', 'Р': 'P'
    }
    for cyr, lat in replacements.items():
        text = text.replace(cyr, lat)
    return text.strip()

def get_mol_safe(query, try_aromatic=False):
    q = query
    if try_aromatic:
        q = query.lower()
    
    mol = Chem.MolFromSmiles(q)
    if mol: return mol, q
    
    path = chem_builder.resolve_input_to_file(query)
    if path:
        try:
            if path.lower().endswith('.sdf'):
                m = Chem.MolFromMolFile(path)
            else:
                m = Chem.MolFromPDBFile(path)
            if m: return m, os.path.basename(path)
        except: pass
    
    return None, None

#def analyze_molecule(target):
#    mol, name = get_mol_safe(target)
#    if not mol: return f"Error: '{target}' not found or invalid."
#    
#    try:
#        mw = Descriptors.MolWt(mol)
#        logp = Descriptors.MolLogP(mol)
#        return f"\n--- REPORT: {name} ---\nWeight: {mw:.2f} | LogP: {logp:.2f}"
#    except: return "Error: Could not calculate descriptors."

def search_similarity(query_input, top_n=5, method="morgan"):
    mols = GLOBAL_MOLS

    clean_q = fix_chemical_input(query_input.strip().strip('"').strip("'"))
    
    name_map = {
        "BENZENE": "c1ccccc1", "BENZOL": "c1ccccc1",
        "PHENOL": "Oc1ccccc1", "ASPIRIN": "CC(=O)Oc1ccccc1C(=O)O",
        "PYRIDINE": "c1ncccc1"
    }
    if clean_q.upper() in name_map: clean_q = name_map[clean_q.upper()]

    mol, source_name = get_mol_safe(clean_q)
    if not mol: return "Error: Invalid molecule source."
    
    lfc = rdMolStandardize.LargestFragmentChooser()
    un = rdMolStandardize.Uncharger()

    try:
        mol = lfc.choose(mol) # - солі
        mol = un.uncharge(mol) # нейтралізація
    except Exception:
        pass

    print(f"   🧬 [RDKit] Generating {method.upper()} fingerprints for similarity search...")
    
    try:
        if method == "morgan":
            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        elif method == "maccs":
            query_fp = MACCSkeys.GenMACCSKeys(mol)
        else: 
            query_fp = Chem.PatternFingerprint(mol)
    except: return "Error: Could not generate query fingerprint."

    db = chembl_db.load_db()
    results = []
    
    for entry in db:
        target_mol = entry.get('mol')
        if not target_mol: continue
            
        try:
            if method == "morgan":
                target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=1024)
                score = DataStructs.TanimotoSimilarity(query_fp, target_fp)
            elif method == "maccs":
                target_fp = MACCSkeys.GenMACCSKeys(target_mol)
                score = DataStructs.TanimotoSimilarity(query_fp, target_fp)
            else: 
                target_fp = entry.get('fp') 
                if not target_fp: target_fp = Chem.PatternFingerprint(target_mol)
                score = DataStructs.DiceSimilarity(query_fp, target_fp)
                
            results.append((score, entry.get('name', 'Unknown'), target_mol))
        except: continue
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    grouped_results = {}
    
    for score, name, t_mol in results[:100]: 
        try:
            clean_mol = lfc.choose(t_mol)
            clean_mol = un.uncharge(clean_mol)
            
            ikey = Chem.MolToInchiKey(clean_mol)
            skeleton = ikey.split('-')[0] if ikey else name
        except:
            skeleton = name 
            
        if skeleton not in grouped_results:
            grouped_results[skeleton] = {'score': score, 'names': [name]}
        else:
            if name not in grouped_results[skeleton]['names']:
                grouped_results[skeleton]['names'].append(name)

    final_groups = list(grouped_results.values())[:top_n]

    report = f"\n--- SIMILARITY FOR: {source_name} (Method: {method.upper()}) ---\n"
    for i, group in enumerate(final_groups):
        names_str = ", ".join(group['names'])
        report += f"   {i+1}. {names_str} (Score: {group['score']:.3f})\n"
        
    return report

def search_substructure(query_input):
    mols = GLOBAL_MOLS

    clean_q = fix_chemical_input(query_input.strip().strip('"').strip("'"))
    
    name_map = {
        "BENZENE": "c1ccccc1", "BENZOL": "c1ccccc1",
        "PYRIDINE": "c1ccncc1", "QUINOLINE": "c1ncccc1",
        "INDOLE": "c1ccc2c(c1)cc[nH]2"
    }
    
    replacements = {
        "N(=O)=O": "[$([N+](=O)[O-]),$([N](=O)=O)]",
        "S(=O)(=O)": "S(=O)(=O)"
    }

    if clean_q in replacements:
        clean_q = replacements[clean_q]
    elif clean_q.upper() in name_map:
        clean_q = name_map[clean_q.upper()]

    mol = None
    mode = "Unknown"

    temp_mol = Chem.MolFromSmiles(clean_q)
    
    if temp_mol and clean_q.isupper() and "1" in clean_q:
        temp_mol_aro = Chem.MolFromSmiles(clean_q.lower())

    if temp_mol:
        try:
            flexible_smarts = Chem.MolToSmarts(temp_mol)
            mol = Chem.MolFromSmarts(flexible_smarts)
            if mode == "Unknown": mode = f"Auto-Pattern ({clean_q})"
        except: pass

    if not mol:
        mol = Chem.MolFromSmarts(clean_q)
        mode = "Direct SMARTS"
        
    if not mol:
        path = chem_builder.resolve_input_to_file(clean_q)
        if path:
            mode = f"File: {os.path.basename(path)}"
            try:
                if path.lower().endswith('.sdf'):
                    mol = Chem.MolFromMolFile(path)
                    if mol: 
                        try: mol = Chem.RemoveHs(mol)
                        except: pass
                else:
                    mol = Chem.MolFromPDBFile(path)
            except: pass

    if not mol: return f"Error: Invalid query '{clean_q}'."

    db = chembl_db.load_db()
    matches = []
    
    for entry in db:
        try:
            if entry['mol'].HasSubstructMatch(mol):
                matches.append(entry['name'])
        except: continue
        
    if not matches: return f"No matches found via {mode}."
    
    report = f"\n--- SUBSTRUCTURE MATCHES ({mode}) ---\nFound {len(matches)} matches: "
    report += ", ".join(matches[:15])
    if len(matches) > 15: report += f"... and {len(matches)-15} more."
    return report

def cluster_db(cutoff=0.6, method="morgan"):

    mols = GLOBAL_MOLS

    if not mols:
        return "Error: Database is empty."
    
    if method == "scaffold":
        print("   🧪 [RDKit] Extracting Bemis-Murcko Scaffolds...")
        scaffold_groups = defaultdict(list)
        
        db = chembl_db.load_db() 
        
        for entry in db:
            m = entry.get('mol')
            drug_name = entry.get('name', 'Unknown')
            
            if m is None: continue
            try:
                scaff_smiles = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
                
                if not scaff_smiles: 
                    scaff_smiles = "Acyclic/Linear (No rings)"
                    
                scaffold_groups[scaff_smiles].append(drug_name) 
            except Exception:
                continue
                
        report = f"--- SCAFFOLD CLUSTERING REPORT ---\n"
        report += f"Total Unique Cores: {len(scaffold_groups)}\n"
        
        sorted_scaffs = sorted(scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (scaff, group) in enumerate(sorted_scaffs[:10], 1):
            examples = ", ".join(group[:3]) 
            report += f"   Core {i}: {len(group)} drugs. Scaffold: {scaff}\n"
            report += f"            Examples: {examples}\n"
            
        return report

    dist_cutoff = 1.0 - cutoff 
    
    if method == "maccs":
        db = chembl_db.load_db()
        fps = [x['fp'] for x in db]
        dists = []
        npts = len(fps)
        for i in range(1, npts):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1-x for x in sims])

        clusters = Butina.ClusterData(dists, npts, dist_cutoff, isDistData=True) 
        
        report = f"\n--- CLUSTERING REPORT (Similarity > {cutoff}) ---\nTotal Clusters: {len(clusters)}\n"
        for idx, cluster in enumerate(clusters[:10]):
            centroid_name = db[cluster[0]]['name']
            report += f"   Cluster {idx+1}: {len(cluster)} drugs. (Centroid: {centroid_name})\n"
        return report
    
    if method == "hierarchical":
        print(f"   🧬 [RDKit & SciPy] Performing Hierarchical Clustering (Morgan FPs, Average Linkage)...")
        
        db = chembl_db.load_db()
        valid_mols = []
        valid_names = []
        for entry in db:
            if entry.get('mol') is not None:
                valid_mols.append(entry['mol'])
                valid_names.append(entry.get('name', 'Unknown'))
                
        if not valid_mols:
            return "Error: No valid molecules found for clustering."

        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in valid_mols]
        
        fps_array = np.zeros((len(fps), 1024), dtype=int)
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, fps_array[i])
            
        dist_matrix = pdist(fps_array, metric='jaccard')
        
        Z = linkage(dist_matrix, method='average')
        
        dist_threshold = 1.0 - cutoff
        cluster_labels = fcluster(Z, t=dist_threshold, criterion='distance')
        
        clusters = {}
        for idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(valid_names[idx])
            
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"   📊 [SciPy & Matplotlib] Generating Dendrogram...")
        plt.figure(figsize=(12, 8))
        plt.title('Hierarchical Clustering Dendrogram of Chemical Space')
        plt.xlabel('Cluster Size (or Molecule Index if < 30)')
        plt.ylabel('Jaccard Distance (1 - Tanimoto Similarity)')

        dendrogram(
            Z,
            truncate_mode='lastp',
            p=30,
            leaf_rotation=90.,
            leaf_font_size=10.,
            show_contracted=True,
            color_threshold=dist_threshold
        )
        plt.ylim(0.9, 1.00)
        
        results_dir = "results"
        plot_filename = "hierarchical_dendrogram.png"
        plot_path = os.path.join(results_dir, plot_filename)
        
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        report = f"--- HIERARCHICAL CLUSTERING REPORT (Morgan ECFP4, Sim > {cutoff}) ---\n"
        report += f"Total Clusters: {len(sorted_clusters)}\n"
        for i, (cid, members) in enumerate(sorted_clusters[:10]): 
            centroid = members[0] 
            report += f"   Cluster {i+1}: {len(members)} drugs. (Example: {centroid})\n"
            
        report += f"\n[Visualization]: Dendrogram saved successfully as '{plot_path}'."

        return report
    return f"Error: Unknown clustering method '{method}'. Available: scaffold, maccs, hierarchical."