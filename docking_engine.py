import os
import re
import subprocess
import urllib.request
import shutil
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
import chem_research
import json
from Bio.PDB import PDBParser, Superimposer, PDBIO

def get_ensemble_receptors(base_pdb, limit=5):
    base_pdb = base_pdb.upper()
    print(f"   🔍 [PDB API] Searching for conformation ensemble for {base_pdb}...")
    try:
        url_info = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{base_pdb}/1"
        req = urllib.request.Request(url_info)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            uniprot_id = data["rcsb_polymer_entity_container_identifiers"]["reference_sequence_identifiers"][0]["database_accession"]
            
        print(f"   🧬 [UniProt] Found protein ID: {uniprot_id}. Searching for the best crystals...")
        
        query = {
          "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
              "operator": "exact_match",
              "value": uniprot_id
            }
          },
          "request_options": {
            "paginate": {"start": 0, "rows": 20},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]
          },
          "return_type": "entry"
        }
        
        req_search = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", 
                                            data=json.dumps(query).encode('utf-8'),
                                            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req_search) as response:
            search_data = json.loads(response.read().decode())
            
        found_pdbs = [item["identifier"] for item in search_data["result_set"]]
        ensemble = [base_pdb]
        
        for p in found_pdbs:
            if p != base_pdb and len(ensemble) < limit:
                ensemble.append(p)
                
        print(f"   🏆 [Ensemble] Selected structures for docking: {', '.join(ensemble)}")
        return ensemble
        
    except Exception as e:
        print(f"   ⚠️ [API Error] Failed to collect ensemble ({e}). Using only {base_pdb}.")
        return [base_pdb]

def get_protein_bbox(pdb_file):
    x, y, z = [], [], []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x.append(float(line[30:38]))
                y.append(float(line[38:46]))
                z.append(float(line[46:54]))
    
    if not x: return None, None
    
    center = [round((max(x) + min(x)) / 2, 3), 
              round((max(y) + min(y)) / 2, 3), 
              round((max(z) + min(z)) / 2, 3)]
    
    size = [round((max(x) - min(x)) + 15, 3),
            round((max(y) - min(y)) + 15, 3),
            round((max(z) - min(z)) + 15, 3)]
    
    size = [min(s, 80.0) for s in size]
    return center, size

def generate_flex_receptor(receptor_pdbqt, flex_res):
    rigid_out = receptor_pdbqt.replace(".pdbqt", "_rigid.pdbqt")
    flex_out = receptor_pdbqt.replace(".pdbqt", "_flex.pdbqt")

    abs_receptor_pdbqt = os.path.abspath(receptor_pdbqt)
    abs_rigid_out = os.path.abspath(rigid_out)
    abs_flex_out = os.path.abspath(flex_out)

    molname = os.path.basename(receptor_pdbqt).replace(".pdbqt", "")
    
    chain_id = "A"
    clean_res = []
    
    for r in flex_res:
        if ":" in r:
            parts = r.split(':')
            chain_id = parts[0]
            clean_res.append(parts[1])
        else:
            clean_res.append(r)
            
    res_joined = "_".join(clean_res)
    flex_str = f"{molname}:{chain_id}:{res_joined}"

    print(f"   ⚙️ [MGLTools] Generating torsion trees for {flex_str}...")
    
    mgl_python = r"C:\Program Files (x86)\MGLTools-1.5.7\python.exe"
    flex_script = r"C:\Program Files (x86)\MGLTools-1.5.7\Lib\site-packages\AutoDockTools\Utilities24\prepare_flexreceptor4.py"
    
    cmd = [
        mgl_python, 
        flex_script,
        "-r", abs_receptor_pdbqt, 
        "-s", flex_str, 
        "-g", abs_rigid_out, 
        "-x", abs_flex_out
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(abs_rigid_out) and os.path.exists(abs_flex_out):
            if os.path.getsize(abs_flex_out) < 50:
                print(f"   ⚠️ [Error] MGLTools created an empty flex file. Residues {flex_str} not found!")
                return None, None
            
            return abs_rigid_out, abs_flex_out
        else:
            error_msg = result.stderr.strip() or result.stdout.strip()
            print(f"   ⚠️ [Flex Script Error] {error_msg}")
            return None, None
    except Exception as e:
        print(f"   ⚠️ [CRITICAL ERROR] {str(e)}")
        return None, None

def align_proteins(ref_pdb, mobile_pdb, aligned_pdb):
    print(f"   📐 [Alignment] Superimposing {os.path.basename(mobile_pdb)} onto reference {os.path.basename(ref_pdb)}...")
    try:
        from Bio.PDB import PDBParser, Superimposer, PDBIO
        from Bio import pairwise2
        
        aa_3to1 = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }

        parser = PDBParser(QUIET=True)
        ref_struct = parser.get_structure("ref", ref_pdb)
        mob_struct = parser.get_structure("mob", mobile_pdb)

        def get_longest_chain_ca(struct):
            best_chain = None
            max_res = 0
            for model in struct:
                for chain in model:
                    aa_count = sum(1 for res in chain if res.get_resname() in aa_3to1 and 'CA' in res)
                    if aa_count > max_res:
                        max_res = aa_count
                        best_chain = chain
            
            seq = ""
            ca_atoms = []
            if best_chain:
                for res in best_chain:
                    resname = res.get_resname()
                    if resname in aa_3to1 and 'CA' in res:
                        seq += aa_3to1[resname]
                        ca_atoms.append(res['CA'])
            return seq, ca_atoms

        ref_seq, ref_ca = get_longest_chain_ca(ref_struct)
        mob_seq, mob_ca = get_longest_chain_ca(mob_struct)

        if not ref_ca or not mob_ca:
            print("   ⚠️ [Alignment Warning] Missing CA atoms or unrecognized residues.")
            return False

        alignments = pairwise2.align.globalxx(ref_seq, mob_seq)
        if not alignments:
            return False
            
        best_alignment = alignments[0]
        ref_aligned, mob_aligned, _, _, _ = best_alignment
        
        matched_ref = []
        matched_mob = []
        
        ref_idx = 0
        mob_idx = 0
        
        for r_char, m_char in zip(ref_aligned, mob_aligned):
            if r_char != '-' and m_char != '-':
                if r_char == m_char: 
                    matched_ref.append(ref_ca[ref_idx])
                    matched_mob.append(mob_ca[mob_idx])
            
            if r_char != '-': ref_idx += 1
            if m_char != '-': mob_idx += 1

        print(f"   🧬 [Alignment] Matched {len(matched_ref)} homologous CA atoms.")
        
        if len(matched_ref) < 20:
            print("   ⚠️ [Alignment Warning] Not enough matching residues.")
            return False

        super_imposer = Superimposer()
        super_imposer.set_atoms(matched_ref, matched_mob)
        super_imposer.apply(mob_struct.get_atoms())

        io = PDBIO()
        io.set_structure(mob_struct)
        io.save(aligned_pdb)
        print(f"   ✅ [Alignment] RMSD: {super_imposer.rms:.3f} Å")
        return True
    except Exception as e:
        print(f"   ⚠️ [Alignment Error] {e}")
        return False

def perform_docking(target_input, receptor=None, flex_res=None, reference_receptor=None):
    if receptor is None:
        receptor = "6LU7"
        
    receptor_id = receptor.upper()
    
    mol, mol_name = chem_research.get_mol_safe(target_input)
    if not mol: return f"Error: Could not find molecule '{target_input}'"
    
    clean_mol_name = mol_name.replace(".sdf", "").replace(".mol", "")
    
    import re
    import hashlib
    if re.search(r'[=()#@:\[\]\\]', clean_mol_name):
        short_hash = hashlib.md5(clean_mol_name.encode()).hexdigest()[:6]
        clean_mol_name = f"DeNovo_{short_hash}"
        print(f"   🛡️ [Sanitizer] Raw SMILES detected. Safe name generated: {clean_mol_name}")
    
    base_rec = reference_receptor.upper() if reference_receptor else receptor_id
    results_dir = os.path.join("Docking_Results", f"{clean_mol_name}_{base_rec}")
    os.makedirs(results_dir, exist_ok=True)
    
    base_rec = reference_receptor.upper() if reference_receptor else receptor_id
    results_dir = os.path.join("Docking_Results", f"{clean_mol_name}_{base_rec}")
    os.makedirs(results_dir, exist_ok=True)
    
    pdb_file = os.path.join(results_dir, f"{receptor_id}.pdb")
    pdbqt_file = os.path.join(results_dir, f"{receptor_id}_clean.pdbqt")
    
    if not os.path.exists(pdbqt_file):
        print(f"   🧬 [Cloud] Downloading protein {receptor_id}...")
        try:
            url = f"https://files.rcsb.org/download/{receptor_id}.pdb"
            urllib.request.urlretrieve(url, pdb_file)
            
            if reference_receptor and receptor_id != reference_receptor.upper():
                ref_id = reference_receptor.upper()
                ref_pdb = os.path.join(results_dir, f"{ref_id}.pdb")
                
                if not os.path.exists(ref_pdb):
                    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{ref_id}.pdb", ref_pdb)
                
                aligned_pdb = os.path.join(results_dir, f"{receptor_id}_aligned.pdb")
                is_aligned = align_proteins(ref_pdb, pdb_file, aligned_pdb)
                
                if is_aligned:
                    pdb_file = aligned_pdb 
            
            print(f"   🧹 [Cleanup] Basic PDB preparation (removing water, HETATM, and alt locations)...")
            clean_pdb_file = os.path.join(results_dir, f"{receptor_id}_stripped.pdb")
            with open(pdb_file, 'r') as f_in, open(clean_pdb_file, 'w') as f_out:
                for line in f_in:
                    if line.startswith("CONECT") or line.startswith("HETATM"):
                        continue
                    
                    if line.startswith("ATOM"):
                        alt_loc = line[16]
                        if alt_loc not in [' ', 'A', '1']:
                            continue
                        line = line[:16] + ' ' + line[17:]
                        
                    f_out.write(line)
            
            print(f"   ⚙️ [MGLTools] Preparing PDBQT (preserving chains and adding hydrogens)...")
            
            mgl_python = r"C:\Program Files (x86)\MGLTools-1.5.7\python.exe"
            prep_rec_script = r"C:\Program Files (x86)\MGLTools-1.5.7\Lib\site-packages\AutoDockTools\Utilities24\prepare_receptor4.py"
            
            abs_clean_pdb = os.path.abspath(clean_pdb_file)
            abs_pdbqt = os.path.abspath(pdbqt_file)
            
            cmd_prep = [
                mgl_python,
                prep_rec_script,
                "-r", abs_clean_pdb,
                "-o", abs_pdbqt,
                "-A", "hydrogens",
                "-U", "nphs_lps_waters"
            ]
            
            result_prep = subprocess.run(cmd_prep, capture_output=True, text=True)
            
            if not os.path.exists(abs_pdbqt):
                error_msg = result_prep.stderr.strip() or result_prep.stdout.strip()
                print(f"   ❌ [MGLTools Prep Error] Failed to create rigid receptor! Log: {error_msg}")
                return f"Error preparing receptor."
                           
            if os.path.exists(clean_pdb_file):
                os.remove(clean_pdb_file)
                
        except Exception as e:
            return f"Error preparing receptor: {str(e)}"

    receptor_args = ["--receptor", pdbqt_file]
    flex_info = "Rigid"
    
    if flex_res:
        if isinstance(flex_res, str):
            flex_res = [flex_res]
            
        print(f"   🧬 [Flex] Preparing flexible residues: {', '.join(flex_res)}")
        rigid_pdbqt, flex_pdbqt = generate_flex_receptor(pdbqt_file, flex_res)
        
        if rigid_pdbqt and flex_pdbqt:
            receptor_args = ["--receptor", rigid_pdbqt, "--flex", flex_pdbqt]
            flex_info = f"Flexible ({', '.join(flex_res)})"
        else:
            print(f"   ⚠️ [Fallback] Switching to Rigid Docking.")
            flex_info = "Rigid (Flex preparation failed)"

    if reference_receptor:
        ref_pdb_path = os.path.join(results_dir, f"{reference_receptor.upper()}.pdb")
        center, size = get_protein_bbox(ref_pdb_path)
        print(f"   📦 [Grid Box] Locked to reference {reference_receptor.upper()}: Center {center}")
    else:
        center, size = get_protein_bbox(pdb_file)
        print(f"   📦 [Grid Box] Using local coordinates: Center {center}")
    
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        preparator = MoleculePreparation()
        preparator.prepare(mol)
        
        temp_ligand = os.path.join(results_dir, "temp_ligand.pdbqt")
        preparator.write_pdbqt_file(temp_ligand)
        
        out_prefix = os.path.join(results_dir, "temp_ligand_out.pdbqt")

        cmd = [
            "vina.exe",
            *receptor_args,
            "--ligand", temp_ligand,
            "--out", out_prefix,
            "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
            "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
            "--exhaustiveness", "16"
        ]

        print(f"   🎯 [Vina.exe] Simulating interaction for {clean_mol_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        saved_files_msg = ""
        
        if os.path.exists(out_prefix):
            base_filename = f"{receptor_id}_{clean_mol_name}"
            tag = "flex_docked" if "Flexible" in flex_info else "docked"
            
            final_pdbqt = os.path.join(results_dir, f"{base_filename}_{tag}.pdbqt")
            final_sdf = os.path.join(results_dir, f"{base_filename}_{tag}.sdf")
            
            shutil.move(out_prefix, final_pdbqt)
            
            print(f"   🔄 [OpenBabel] Saving 3D coordinates to SDF...")
            subprocess.run(["obabel", "-i", "pdbqt", final_pdbqt, "-o", "sdf", "-O", final_sdf], capture_output=True)
                
            saved_files_msg = f"📁 Files saved in '{results_dir}'\n   - {base_filename}_{tag}.sdf\n   - {os.path.basename(pdbqt_file)}"
        
        if os.path.exists(temp_ligand): os.remove(temp_ligand)

        affinity = None
        for line in result.stdout.split('\n'):
            if re.match(r'^\s*1\s+[-0-9.]+', line):
                affinity = float(line.split()[1])
                break

        if affinity is None:
            return f"Error in simulation. Vina output: {result.stderr}"

        report = f"\n{'='*40}\n"
        report += f"🧪 MOLECULAR DOCKING REPORT\n"
        report += f"{'='*40}\n"
        report += f"Ligand:   {mol_name}\n"
        report += f"Protein:  {receptor_id}\n"
        report += f"Mode:     {flex_info}\n"
        report += f"Affinity: {affinity:.2f} kcal/mol\n"
        if saved_files_msg:
            report += f"{'-'*40}\n{saved_files_msg}\n"
        report += f"{'='*40}\n"
        
        return report

    except Exception as e:
        return f"Docking Critical Error: {str(e)}"