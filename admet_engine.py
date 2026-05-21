from admet_ai import ADMETModel
import chem_research
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
os.environ["PYTORCH_LIGHTNING_SUPPRESS_WARNINGS"] = "1"

print("⏳ Loading Deep Learning ADMET Models (admet_ai)...")
admet_model = ADMETModel()
print("✅ ML Models Ready!")

def predict_admet(target, requested_params=None):
    mol, name = chem_research.get_mol_safe(target)
    if not mol: 
        return f"Error: '{target}' not found or invalid."
    
    try:
        rdkit_mw = Descriptors.MolWt(mol)
        rdkit_logp = Descriptors.MolLogP(mol)
        rdkit_tpsa = Descriptors.TPSA(mol)

        smiles = Chem.MolToSmiles(mol)
        preds_df = admet_model.predict(smiles=[smiles])
        preds = preds_df.iloc[0].to_dict()
        
        bbb_prob = preds.get('BBB_Martins', 0)
        bbb_ml = 'Positive' if bbb_prob > 0.5 else 'Negative'
        
        hia_prob = preds.get('HIA_Hou', 0)
        hia_ml = 'High' if hia_prob > 0.5 else 'Low'

        metrics = {
            "MW": f"Molecular Weight (MW): {rdkit_mw:.2f} g/mol (RDKit Exact)",
            "LogP": f"Lipophilicity (LogP): {rdkit_logp:.2f} (RDKit Crippen Method)",
            "TPSA": f"Polar Surface Area (TPSA): {rdkit_tpsa:.2f} Å² (RDKit Geo)",
            
            "HIA": f"Intestinal Absorption (HIA): {'High' if preds.get('HIA_Hou', 0) > 0.5 else 'Low'} (Prob: {preds.get('HIA_Hou', 0):.2f})",
            "Caco2": f"Caco-2 Permeability: {preds.get('Caco2_Wang', 0):.2f} (optimal > -5.15)",
            "BBB": f"BBB Penetration: {'Positive' if preds.get('BBB_Martins', 0) > 0.5 else 'Negative'} (Prob: {preds.get('BBB_Martins', 0):.2f})",
            "PPBR": f"Plasma Protein Binding: {preds.get('PPBR_AZ', 0):.2f}%",
            "CYP3A4": f"CYP3A4 Inhibition: {'Yes' if preds.get('CYP3A4_Veith', 0) > 0.5 else 'No'}",
            "CYP2D6": f"CYP2D6 Inhibition: {'Yes' if preds.get('CYP2D6_Veith', 0) > 0.5 else 'No'}",
            "Clearance": f"Hepatic Clearance: {preds.get('Clearance_Hepatocyte_AZ', 0):.2f} µL/min/10^6 cells",
            "HalfLife": f"Half-Life: {preds.get('Half_Life_Obach', 0):.2f} hours",
            "hERG": f"hERG Cardiotoxicity: {'High Risk' if preds.get('hERG', 0) > 0.5 else 'Safe'} (Prob: {preds.get('hERG', 0):.2f})",
            "AMES": f"AMES Mutagenicity: {'Positive' if preds.get('AMES', 0) > 0.5 else 'Negative'} (Prob: {preds.get('AMES', 0):.2f})",
            "DILI": f"Liver Injury (DILI): {'High Risk' if preds.get('DILI', 0) > 0.5 else 'Safe'} (Prob: {preds.get('DILI', 0):.2f})"
        }
        
        report = f"\n=== HYBRID ADMET PROFILE: {name} ===\n"
        
        if not requested_params:
            report += f"0. [P] PHYSICOCHEMICAL PROPERTIES (RDKit Math)\n   - {metrics['MW']}\n   - {metrics['LogP']}\n   - {metrics['TPSA']}\n"
            report += f"\n1. [A] ABSORPTION (Всмоктування ML)\n   - {metrics['HIA']}\n   - {metrics['Caco2']}\n"
            report += f"\n2. [D] DISTRIBUTION (Розподіл ML)\n   - {metrics['BBB']}\n   - {metrics['PPBR']}\n"
            report += f"\n3. [M] METABOLISM (Метаболізм ML)\n   - {metrics['CYP3A4']}\n   - {metrics['CYP2D6']}\n"
            report += f"\n4. [E] EXCRETION (Виведення ML)\n   - {metrics['Clearance']}\n   - {metrics['HalfLife']}\n"
            report += f"\n5. [T] TOXICITY (Токсичність ML)\n   - {metrics['hERG']}\n   - {metrics['AMES']}\n   - {metrics['DILI']}\n"
        else:
            report += "\n🎯 Запрошенные параметры:\n"
            report += f"   - {metrics['MW']}\n   - {metrics['LogP']}\n   - {metrics['TPSA']}\n"
            report += "   ----------------------------------\n"
            
            for param in requested_params:
                matched_key = next((k for k in metrics.keys() if k.lower() == param.lower()), None)
                if matched_key and matched_key not in ["MW", "LogP", "TPSA"]:
                    report += f"   - {metrics[matched_key]}\n"
                elif not matched_key:
                    report += f"   - {param}: ⚠️ Параметр не найден\n"
        
        report += f"\n[Scientific Insight]: Toxicity/ADME powered by D-MPNN (TDC). PhysChem descriptors calculated strictly via RDKit."
        
        return report
    except Exception as e:
        return f"Hybrid ADMET Error: {str(e)}"