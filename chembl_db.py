import os
import pickle
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

DB_FILE = "drugs_db.pkl"

def download_and_build_db():
    print("🌍 Connecting to ChEMBL... Downloading approved drugs list.")
    
    molecule = new_client.molecule
    res = molecule.filter(
        max_phase=4, 
        molecule_type='Small molecule', 
        molecule_properties__mw_freebase__lte=600 
    ).only(['molecule_chembl_id', 'pref_name', 'molecule_structures'])

    print(f"📦 Molecules found: {len(res)}. Starting download and processing...")

    data = []
    for item in tqdm(res):
        try:
            name = item['pref_name']
            chembl_id = item['molecule_chembl_id']
            if not item['molecule_structures']: continue
            smiles = item['molecule_structures']['canonical_smiles']
            
            if not smiles or not name: continue

            mol = Chem.MolFromSmiles(smiles)
            if not mol: continue

            fp = Chem.PatternFingerprint(mol)

            data.append({
                'id': chembl_id,
                'name': name,
                'smiles': smiles,
                'fp': fp,
                'mol': mol
            })
        except Exception:
            continue

    print(f"💾 Saving archive '{DB_FILE}'...")
    with open(DB_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Done! Database contains {len(data)} drugs.")

def load_db():
    if not os.path.exists(DB_FILE):
        download_and_build_db()
    
    with open(DB_FILE, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    download_and_build_db()