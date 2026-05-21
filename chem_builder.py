import os
import warnings
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
import chem_utils

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)

def save_mol(mol, name_hint, output_format):
    output_format = output_format.lower().replace("sdf3", "sdf")
    path = chem_utils.get_unique_filename(name_hint, output_format)
    
    try:
        if "sdf" in output_format or "mol" in output_format:
            w = Chem.SDWriter(path)
            w.write(mol)
            w.close()
        elif "pdb" in output_format:
            Chem.MolToPDBFile(mol, path)
        else:
            return None
        return path
    except:
        return None

def create_molecule_file(input_str, output_format, forced_name=None):
    input_str = input_str.strip('"').strip("'")
    mol = Chem.MolFromSmiles(input_str)
    
    if mol:
        name_hint = forced_name if forced_name else "custom_mol"
    else:
        try:
            compounds = pcp.get_compounds(input_str, 'name')
            if not compounds: return f"Error: '{input_str}' not found."
            mol = Chem.MolFromSmiles(compounds[0].isomeric_smiles)
            name_hint = input_str
        except Exception as e:
            return f"Error: {e}"

    if not mol: return "Error: Build failed."
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    res_path = save_mol(mol, name_hint, output_format)
    if res_path:
        return f"Success: Created '{os.path.basename(res_path)}' in {chem_utils.RESULTS_DIR}"
    return "Error: Could not save file."

def resolve_input_to_file(query):
    existing = chem_utils.find_file_smart(query)
    if existing: return existing
    
    query_clean = query.strip().strip('"').strip("'")
    if "." in query_clean: query_clean = os.path.splitext(query_clean)[0]
    
    msg = create_molecule_file(query_clean, "sdf", forced_name=query_clean)
    if "Success" in msg:
        return chem_utils.find_file_smart(query_clean)
    return None

def convert_local_file(input_filename, target_format):
    real_filename = resolve_input_to_file(input_filename)
    if not real_filename: return f"Error: '{input_filename}' not found."

    ext = real_filename.split('.')[-1].lower()
    mol = None
    try:
        if ext in ['sdf', 'mol']:
            suppl = Chem.SDMolSupplier(real_filename)
            mol = suppl[0] if len(suppl) > 0 else None
        elif ext == 'pdb':
            mol = Chem.MolFromPDBFile(real_filename)
        
        if mol is None: return f"Error: Failed to parse {real_filename}"
        base_name = os.path.splitext(os.path.basename(real_filename))[0]
        res = save_mol(mol, base_name, target_format)
        return f"Success: Converted to '{os.path.basename(res)}'" if res else "Error."
    except Exception as e:
        return f"Error: {e}"