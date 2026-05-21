import chem_builder
import chem_research
import docking_engine
import os

def create_molecule_file(s, f, n=None): return chem_builder.create_molecule_file(s, f, n)
#def analyze_molecule(t): return chem_research.analyze_molecule(t)
def search_similarity(q, top_n=5, method="morgan"): return chem_research.search_similarity(q, top_n=top_n, method=method)
def search_substructure(q): return chem_research.search_substructure(q)
def cluster_db(cutoff=0.6, method="morgan"): return chem_research.cluster_db(cutoff, method)

def predict_admet(target, requested_params=None):
    import admet_engine
    return admet_engine.predict_admet(target, requested_params=requested_params)

def perform_docking(ligand, receptor=None, flex_res=None, reference_receptor=None): 
    import docking_engine
    return docking_engine.perform_docking(ligand, receptor=receptor, flex_res=flex_res, reference_receptor=reference_receptor)

def get_ensemble_receptors(base_pdb, limit=3):
    import docking_engine
    return docking_engine.get_ensemble_receptors(base_pdb, limit=limit)

def convert_local_file(f, t):
    return chem_builder.convert_local_file(f, t)