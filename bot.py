from openai import OpenAI
import tools
import sys, io
import warnings
import json
import re
import time
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', line_buffering=True)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

system_instruction = """
You are a Chemoinformatics JSON Agent.
Map user queries to the correct tool and target. You MUST return ONLY a JSON object.

CRITICAL RULE 0: YOU ARE A NON-CONVERSATIONAL ENGINE.
You are a backend API, not a chatbot. NEVER use greetings, NEVER explain your actions, and NEVER answer questions yourself. Your ONLY possible output is a single JSON object. If you output plain text, the system will crash.

CRITICAL RULE 1: STRICT TOOL SELECTION.
You MUST select a tool ONLY from this exact list: SEARCH_SUB, SEARCH_SIM, GENERATE, BUILD, CLUSTER, ADMET, DOCK. 
DO NOT invent new tools (e.g., NEVER use "HISTORY", "INFO"). If the user asks for historical, descriptive, or medical text, you MUST use the "ADMET" tool with an empty params array [].

CRITICAL RULE 2: TARGET EXTRACTION.
Never leave the target empty if a drug is mentioned or implied (e.g., "like Caffeine" -> target: "Caffeine").

STRICT INSTRUCTION: Do not use your internal knowledge to describe drugs. If a user asks for properties, safety, or ADMET, you MUST generate a JSON call to the ADMET tool. Even if you know the answer, the tool is the only source of truth.
If a user mentions ANY specific property (e.g., "hERG", "Half-Life", "LogP", "Weight", "Toxicity", "DILI"), you MUST put it into the "params" array. 
Example: "toxicity and weight" -> "params": ["Toxicity", "MW"]
If the user asks a general question like "What is X?", you MUST still generate a JSON call to ADMET. DO NOT ANSWER DIRECTLY.
CROSS-LINGUAL RULE: You MUST apply this exact same logic, tool selection, and JSON formatting to user queries in ANY language (e.g., Russian, Ukrainian, English). Always internally translate the user's intent to the corresponding tool, translate property requests to English "params" (e.g. "токсичність" -> "Toxicity"), and normalize the 'target' to its official English name.
GENERATE vs SEARCH_SIM: If the user asks to "find" or "search" analogs, use SEARCH_SIM. If the user explicitly asks to "generate", "design", "create", or "invent" molecules/analogs, you MUST use GENERATE and extract the "constraints" (like min_sim, max_mw, max_logp).

TOOLS:
- SEARCH_SUB: To find molecules CONTAINING a fragment, ring, or SMILES. The target MUST be a strictly valid SMARTS or SMILES string (e.g., "c1ccccc1", "F", or "[F]"). NEVER output a list of drug names.
- SEARCH_SIM: To find EXISTING analogs or similar drugs already in the database.
- GENERATE: To design, invent, create, or generate completely NEW molecules based on a target seed.
- BUILD: For generating/downloading 3D SDF files ONLY. DO NOT use this tool to calculate properties like LogP, MW, or TPSA.
- CLUSTER: For grouping the database (target is empty "").
- ADMET: For 'ADMET', 'pharmacokinetics', 'toxicity', 'safety', 'ML analysis', 'DILI', 'hERG', 'BBB', AND all physicochemical properties ('LogP', 'MW', 'TPSA', 'Weight').
- DOCK: Docking simulation. Can accept 'receptor' (PDB ID) and 'flex_res' (a list of flexible residues, e.g. ["HIS41", "CYS145"]).

JSON FORMAT:
{
  "reasoning": "Brief logic in English.",
  "tool": "TOOL_NAME",
  "target": "Exact SMILES or Name. For multiple molecules: [\"Aspirin\", \"Ibuprofen\"]. For a folder: \"DIR:Ligands\"",
  "receptor": "1IEP",
  "ensemble": true, // Optional. Use ONLY for DOCK if user asks for ensemble or multiple conformations.
  "params": ["hERG", "LogP"] // ALLOWED VALUES ONLY: "MW", "LogP", "TPSA", "HIA", "Caco-2", "BBB", "PPB", "CYP3A4", "CYP2D6", "Clearance", "Half-Life", "hERG", "AMES", "DILI".
  "flex_res": ["RES1", "RES2"] // Optional, only if the user requests 'flexible' docking
  "top_n": 5, // For SEARCH_SIM or GENERATE
  "constraints": {"min_sim": 0.5, "max_mw": 500, "max_logp": 5.0, "max_attempts": 7, "min_qed": 0.5}, // Optional. Use ONLY for GENERATE.
  "sim_method": "morgan", // Optional. Use ONLY for SEARCH_SIM ("morgan", "maccs", "pattern"). Default is "morgan",
  "cutoff": 0.6, // Optional. Use ONLY for CLUSTER,
  "method": "hierarchical" // Optional. Use ONLY for CLUSTER ("hierarchical", "scaffold", "maccs").
}

STRICT RULES FOR 'target':
1. SMILES CASE SENSITIVITY: Copy SMILES exactly as written (e.g., C1CCCCC1 is aliphatic, c1ccccc1 is aromatic). Never alter capitalization.
2. NAME NORMALIZATION: Translate ANY foreign or trivial names into official generic English names (e.g., "Витамин С" -> "Ascorbic acid", "Парацетамол" -> "Acetaminophen", "Силденафил" -> "Sildenafil").
3. NO FORMULAS: NEVER output chemical formulas (like C8H9NO2 or C9H8O4). The target MUST be either an English text name or a SMILES string.
4. NO INVENTING SMILES: DO NOT generate or hallucinate SMILES from your memory! If the user types a drug name, return the NAME. If they type a fragment (like S(=O)(=O)), return EXACTLY that fragment string.
5. NO DIRECT ANSWERS: NEVER answer the user's chemistry question directly. Your ONLY job is to output the JSON tool command.
6. INTENT DISAMBIGUATION: If the user mentions 'similarity', 'analogs', or 'looks like', ALWAYS use SEARCH_SIM. If they mention 'contains', 'has', 'ring', or 'fragment', use SEARCH_SUB.
7. NO INVENTING OR DERIVATIZING: 
   - Do not add atoms or groups that were not mentioned. 
   - "Benzene" (бензол) is a C6H6 ring. 
   - "Benzoic acid" is a different molecule. 

EXAMPLES:
User: "Find all drugs containing a pyridine ring"
{"reasoning": "User wants molecules containing a pyridine substructure.", "tool": "SEARCH_SUB", "target": "Pyridine"}

User: "Find 3 analogs of Aspirin using MACCS keys"
{"reasoning": "User specified 3 analogs and MACCS method.", "tool": "SEARCH_SIM", "target": "Aspirin", "top_n": 3, "sim_method": "maccs"}

User: "Analyze the pharmacokinetics for Sildenafil"
{"reasoning": "User requested pharmacokinetics. Routing to ADMET and translating name.", "tool": "ADMET", "target": "Sildenafil"}

User: "Get a toxicity forecast for Paracetamol"
{"reasoning": "User requested Toxicity. Routing to ADMET and translating name.", "tool": "ADMET", "target": "Acetaminophen"}

User: "Perform hierarchical database clustering."
{"reasoning": "User wants hierarchical clustering.", "tool": "CLUSTER", "target": "hierarchical", "method": "hierarchical"}

User: "Docking of the 6LU7 protein and the Ibuprofen molecule"
{"reasoning": "User requested blind docking for Ibuprofen against protein 6LU7.", "tool": "DOCK", "target": "Ibuprofen", "receptor": "6LU7"}

User: "Perform flexible docking of 6LU7 and Nirmatrelvir, using HIS41 and CYS145"
{"reasoning": "User requested flexible docking...", "tool": "DOCK", "target": "Nirmatrelvir", "receptor": "6LU7", "flex_res": ["HIS41", "CYS145"]}

User: "Generate 3 Ibuprofen-based molecules with a similarity of at least 0.7"
{"reasoning": "User explicitly asked to generate new molecules with a similarity constraint.", "tool": "GENERATE", "target": "Ibuprofen", "top_n": 3, "constraints": {"min_sim": 0.7}}

User: "Generate 5 analogs of Aspirin with similarity > 0.5, MW under 250, LogP less than 3 and QED > 0.6 using 15 attempts."
{"reasoning": "User wants de novo generation with multiple constraints.", "tool": "GENERATE", "target": "Aspirin", "top_n": 5, "constraints": {"min_sim": 0.5, "max_mw": 250.0, "max_logp": 3.0, "min_qed": 0.6, "max_attempts": 15}}

User: "Dock all ligands from the Ligands folder with the 6LU7 protein"
{"reasoning": "User requested batch docking from a specific folder.", "tool": "DOCK", "target": "DIR:Ligands", "receptor": "6LU7"}

User: "Perform ensemble docking of Nirmatrelvir with 6LU7"
{"reasoning": "User requested ensemble docking. Enabling ensemble mode to find multiple conformations.", "tool": "DOCK", "target": "Nirmatrelvir", "receptor": "6LU7", "ensemble": true}
"""

print("🚀 CADD AI Agent (High-Speed & High-Accuracy Mode) is ready.")

def execute_tool(tool_name, target, cutoff=0.6, method="morgan"):
    tool_name = str(tool_name).strip().upper().replace(" ", "")
    
    if not target and tool_name not in ["CLUSTER"]:
        return "❌ Error: AI could not determine the target molecule from the query."
        
    if "SEARCH" in tool_name and "_" not in tool_name:
        tool_name = tool_name.replace("SEARCH", "SEARCH_")
    if tool_name == "SEARCH_PYRIDINE":
        tool_name = "SEARCH_SUB"

    replacements = {'с': 'c', 'С': 'C', 'о': 'o', 'О': 'O', 'н': 'n', 'Н': 'N', 'р': 'p', 'Р': 'P'}
    if target:
        if isinstance(target, list):
            for i in range(len(target)):
                if isinstance(target[i], str):
                    for cyr, lat in replacements.items():
                        target[i] = target[i].replace(cyr, lat)
        elif isinstance(target, str):
            for cyr, lat in replacements.items():
                target = target.replace(cyr, lat)
    
    if isinstance(target, list) and tool_name != "DOCK":
        target = target[0] if len(target) > 0 else ""

    try:
        if tool_name == "SEARCH_SUB":
            return tools.search_substructure(target)
            
        elif tool_name == "SEARCH_SIM":
            top_n = agent_decision.get("top_n", 5)
            sim_method = agent_decision.get("sim_method", "morgan").lower() 
            
            print(f"   🔍 [RDKit] Searching for top {top_n} similar molecules using {sim_method.upper()}...")
            res = tools.search_similarity(target, top_n=top_n, method=sim_method)
            
            if isinstance(res, str) and ("Error" in res or "Invalid" in res):
                print(f"   🔄 [Auto-Fallback] File/Data for '{target}' missing. Triggering BUILD...")
                build_res = tools.create_molecule_file(target, "sdf")
                print(f"   📥 [Build Log]: {build_res}") 
                
                if "Success" in build_res:
                    match = re.search(r"'(.*?)'", build_res)
                    if match:
                        exact_target = match.group(1).replace(".sdf", "")
                        print(f"   🔍 [Auto-Fallback] Retrying search with exact file: {exact_target}")
                        res = tools.search_similarity(exact_target, top_n=top_n, method=sim_method)
                    else:
                        res = tools.search_similarity(target, top_n=top_n, method=sim_method)
                else:
                    res = f"❌ Error: Auto-fallback failed. Could not download '{target}'."
            return res
            
        elif tool_name == "BUILD":
            return tools.create_molecule_file(target, "sdf")
            
        elif tool_name == "DOCK":
            base_receptor_id = agent_decision.get('receptor')
            flex_residues = agent_decision.get('flex_res')
            use_ensemble = agent_decision.get('ensemble', False)
            
            receptors_to_dock = [base_receptor_id]
            if use_ensemble:
                receptors_to_dock = tools.get_ensemble_receptors(base_receptor_id, limit=5)
            
            if isinstance(flex_residues, str):
                flex_residues = [flex_residues]
                
            targets = []
            import os
            import shutil
            
            raw_target = target[0] if isinstance(target, list) and len(target) == 1 else target
            
            if isinstance(raw_target, str) and raw_target.startswith("DIR:"):
                folder_name = raw_target.replace("DIR:", "").strip()
                
                print(f"   📂 [File System] Reading local folder: {folder_name}/")
                if os.path.exists(folder_name) and os.path.isdir(folder_name):
                    for f in os.listdir(folder_name):
                        if f.endswith((".sdf", ".mol", ".pdbqt")):
                            src = os.path.join(folder_name, f)
                            #if not os.path.exists(f):
                            #    shutil.copy(src, f)
                            #
                            #base_name = os.path.splitext(f)[0]
                            #targets.append(base_name)
                            targets.append(src)
                    
                    if not targets:
                        return f"❌ Error: Folder '{folder_name}' is empty or contains no chemical files (.sdf, .mol)."
                else:
                    return f"❌ Error: Folder '{folder_name}' not found. Please create it and add files."
            else:
                targets = target if isinstance(target, list) else [target]
            
            print(f"   🔍 [Batch Routing] Targets: {len(targets)} molecules, Receptors: {len(receptors_to_dock)} structures, Flex: {flex_residues}")
            
            batch_report = f"\n=== ENSEMBLE/BATCH DOCKING REPORT ===\n"
            
            for rec_id in receptors_to_dock:
                batch_report += f"\n👉 RECEPTOR: {rec_id}\n"
                for t in targets:
                    print(f"   ⚙️ Docking in progress: {t} into {rec_id}...")
                    try:
                        single_res = tools.perform_docking(t, receptor=rec_id, flex_res=flex_residues, reference_receptor=base_receptor_id)
                        
                        affinity_match = re.search(r"Affinity:\s*([-0-9.]+\s*kcal/mol)", single_res)
                        if affinity_match:
                            batch_report += f"   ✅ {t}: {affinity_match.group(1)}\n"
                        else:
                            error_text = single_res.replace('\n', ' | ')
                            batch_report += f"   ⚠️ {t}: Error -> {error_text}\n"
                    except Exception as e:
                        batch_report += f"   ❌ {t}: Calculation error ({str(e)})\n"
                    
            batch_report += "\n========================================\n"
            return batch_report
            
        elif tool_name == "CLUSTER":
            print(f"   🔍 [Routing] Clustering database using method: '{method}' (cutoff: {cutoff})")
            return tools.cluster_db(cutoff=cutoff, method=method)
            
        elif tool_name == "ADMET":
            requested_params = agent_decision.get('params', [])
            if isinstance(requested_params, str):
                requested_params = [requested_params]
            
            smart_params = []
            for p in requested_params:
                p_up = p.upper()
                if "TOX" in p_up or "SAFETY" in p_up:
                    smart_params.extend(["hERG", "AMES", "DILI"])
                elif "PSA" in p_up or "SURFACE" in p_up:
                    smart_params.append("TPSA")
                elif "HALF" in p_up or "LIFE" in p_up:
                    smart_params.append("HalfLife")
                elif "WEIGHT" in p_up or "MASS" in p_up:
                    smart_params.append("MW")
                elif "CLEARANCE" in p_up:
                    smart_params.append("Clearance")
                elif "CACO" in p_up:
                    smart_params.append("Caco2")
                elif "PROTEIN" in p_up or "PPB" in p_up:
                    smart_params.append("PPBR")
                elif "CLEARANCE" in p_up or p_up == "CL":
                    smart_params.append("Clearance")
                else:
                    smart_params.append(p)
            
            requested_params = list(dict.fromkeys(smart_params))

            import chem_research
            from rdkit import Chem
            mol, name = chem_research.get_mol_safe(target)
            if mol:
                target = Chem.MolToSmiles(mol)
                print(f"   🔄 [Auto-Resolve] Converted name '{name}' to SMILES for ADMET.")
            else:
                return f"❌ Error: Could not resolve '{target}' to a valid molecular structure."

            print(f"   🔍 [Routing] Target: {target}, Params: {requested_params}")
            return tools.predict_admet(target, requested_params=requested_params)
            
        elif tool_name == "GENERATE":
            top_n = agent_decision.get("top_n", 5)
            constraints = agent_decision.get("constraints", {})
            min_sim = constraints.get("min_sim", 0.4)
            max_mw = constraints.get("max_mw", 1000.0)
            max_logp = constraints.get("max_logp", 10.0)
            min_qed = constraints.get("min_qed", 0.3)
            
            max_attempts = constraints.get("max_attempts", agent_decision.get("max_attempts", 7))
            
            attempt_match = re.search(r'(\d+)\s*(attempt|попыт|итерац|спроб)', user_input, re.IGNORECASE)
            if attempt_match:
                max_attempts = int(attempt_match.group(1))
            
            print(f"   🧠 [De Novo Design] Generating for {target} (Target: {top_n} molecules)")
            print(f"   📐 [Filters] Sim: >{min_sim}, MW: <{max_mw}, LogP: <{max_logp}, QED: >{min_qed}, Attempts: {max_attempts}")
            
            import chem_research
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs, Descriptors, QED
            
            seed_mol, seed_name = chem_research.get_mol_safe(target)
            if not seed_mol:
                return f"❌ Error: Could not find molecule '{target}'."
            
            seed_smiles = Chem.MolToSmiles(seed_mol)
            seed_fp = AllChem.GetMorganFingerprintAsBitVect(seed_mol, 2, nBits=2048)
            
            valid_novel_mols = []
            seen_smiles = set([seed_smiles])
            
            attempt = 0
            dominant_error_msg = ""
            
            while len(valid_novel_mols) < top_n and attempt < max_attempts:
                attempt += 1
                print(f"   ⏳ [LLM] Designing structures (Attempt {attempt}/{max_attempts})...")
                
                gen_prompt = f"Act as a computational chemist. Modify the SMILES '{seed_smiles}' ({seed_name}) by changing functional groups to create 20 NOVEL, valid SMILES strings. Output ONLY a comma-separated list of SMILES."
                
                if attempt > 1 and dominant_error_msg:
                    print(f"   💡 [Feedback Loop] Sending hint to LLM: {dominant_error_msg}")
                    gen_prompt += f" {dominant_error_msg}"
                
                try:
                    gen_completion = client.chat.completions.create(
                        model="local-model",
                        messages=[{"role": "user", "content": gen_prompt}],
                        temperature=0.7 + (attempt * 0.1),
                        max_tokens=400
                    )
                    raw_smiles = gen_completion.choices[0].message.content.strip().split(',')
                except Exception as e:
                    print(f"   ❌ Generation error: {e}")
                    break
                
                print("   🛡️ [RDKit] Filtering generated structures...")
                
                err_counts = {"invalid": 0, "low_sim": 0, "high_mw": 0, "high_logp": 0, "low_qed": 0}
                
                for s in raw_smiles:
                    s = s.strip()
                    if not s or s in seen_smiles: continue
                    
                    try:
                        mol = Chem.MolFromSmiles(s)
                        if not mol or mol.GetNumAtoms() < 5:
                            err_counts["invalid"] += 1
                            continue
                            
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        sim = DataStructs.TanimotoSimilarity(seed_fp, fp)
                        if sim < min_sim:
                            err_counts["low_sim"] += 1
                            continue
                            
                        mw = Descriptors.MolWt(mol)
                        if mw > max_mw:
                            err_counts["high_mw"] += 1
                            continue
                            
                        logp = Descriptors.MolLogP(mol)
                        if logp > max_logp:
                            err_counts["high_logp"] += 1
                            continue
                            
                        qed_val = QED.qed(mol)
                        if qed_val < min_qed:
                            err_counts["low_qed"] += 1
                            continue
                            
                        valid_novel_mols.append((s, sim, mw, logp, qed_val))
                        seen_smiles.add(s)
                        
                        if len(valid_novel_mols) >= top_n:
                            break 
                    except:
                        err_counts["invalid"] += 1
                        continue
                
                if len(valid_novel_mols) < top_n:
                    feedback_counts = {k: v for k, v in err_counts.items() if k != "invalid"}
                    
                    if any(feedback_counts.values()): 
                        dominant_error = max(feedback_counts, key=feedback_counts.get)
                        
                        if dominant_error == "low_sim":
                            dominant_error_msg = f"NOTE: Your previous molecules were too different. Keep the core closer to {seed_smiles} (similarity must be >= {min_sim})."
                        elif dominant_error == "high_mw":
                            dominant_error_msg = f"NOTE: Your previous molecules were too heavy. Remove large or heavy functional groups to keep MW <= {max_mw}."
                        elif dominant_error == "high_logp":
                            dominant_error_msg = f"NOTE: Your previous molecules were too lipophilic. Add polar groups (like N, O, OH) to keep LogP <= {max_logp}."
                        elif dominant_error == "low_qed":
                            dominant_error_msg = f"NOTE: Your previous molecules had poor drug-likeness. Make them more drug-like (QED >= {min_qed}) by avoiding toxic/reactive fragments and balancing polar surface area."
                    else:
                        dominant_error_msg = ""
            
            valid_novel_mols = sorted(valid_novel_mols, key=lambda x: x[1], reverse=True)
            
            report = f"\n=== 🧬 DE NOVO GENERATION REPORT ===\n"
            report += f"Seed: {seed_name} ({seed_smiles})\n"
            report += f"Constraints: Sim >= {min_sim}, MW <= {max_mw}, LogP <= {max_logp}, QED >= {min_qed}\n\n"
            
            if not valid_novel_mols:
                report += f"❌ After {max_attempts} attempts, the AI failed to create molecules passing the strict filters.\n"
            elif len(valid_novel_mols) < top_n:
                report += f"⚠️ Managed to generate only {len(valid_novel_mols)} valid molecules out of {top_n}:\n"
            else:
                report += f"✅ Successfully generated {top_n} molecules:\n"
                
            for i, (smiles, sim, mw, logp, qed_val) in enumerate(valid_novel_mols, 1):
                clean_smiles = smiles.split('\n')[0].split()[0].strip()
                report += f"{i}. SMILES: {clean_smiles}\n   ├─ Similarity: {sim:.3f}\n   ├─ MW: {mw:.1f} g/mol\n   ├─ LogP: {logp:.2f}\n   └─ QED: {qed_val:.3f}\n"
            report += "===================================="
            
            return report
        else:
            return f"❌ Error: Unknown tool '{tool_name}'."
    except Exception as e:
        return f"❌ Tool Execution Error: {e}"

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]: break
    if user_input.lower() == "clear":
        print("[Memory Cleared]")
        continue
    
    start_time = time.time()
    
    strict_user_prompt = f"{user_input}\n\n[SYSTEM: You MUST respond ONLY with a valid JSON object. Do NOT output any text before or after the JSON. Start your response with {{ and end with }}. CRITICAL: If ANY drug is mentioned in the prompt (e.g. 'like Caffeine', 'for Aspirin', 'у Паксловіду'), you MUST extract its official English name and put it in the 'target' field! NEVER leave 'target' empty if a drug is named.]"

    current_context = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": strict_user_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="local-model",
            messages=current_context,
            temperature=0.0,
            max_tokens=300
        )
        
        raw_res = completion.choices[0].message.content.strip()
        
        json_match = re.search(r'\{.*\}', raw_res, re.DOTALL)
        
        if json_match:
            clean_json = json_match.group(0)
            try:
                agent_decision = json.loads(clean_json)
            except json.JSONDecodeError:
                print(f"❌ Parsing error: LLM returned broken JSON. Response: {clean_json}")
                continue
        else:
            print(f"❌ Error: Model violated the format and returned plain text. Raw response: {raw_res}")
            continue
            
        ai_time = time.time() - start_time

        # =================================================================
        # 🛡️ НЕЙРО-СИМВОЛЬНИЙ MIDDLEWARE (Heuristic Overrides)
        # =================================================================
        user_text = user_input.lower()
        tool = agent_decision.get('tool')
        
        # 1. Жорсткий парсинг методів кластеризації
        if tool == 'CLUSTER':
            if any(w in user_text for w in ['maccs', 'макс']):
                agent_decision['method'] = 'maccs'
            elif any(w in user_text for w in ['scaffold', 'core', 'каркас', 'основ']):
                agent_decision['method'] = 'scaffold'
            else:
                agent_decision['method'] = 'hierarchical' # Default
            print(f"   ⚙️ [Middleware] Set clustering method to: {agent_decision['method']}")

        # 2. Перехоплення назв елементів (SMARTS)
        if tool == 'SEARCH_SUB':
            element_map = {
                'fluorine': '[F]', 'фтор': '[F]', 'f': '[F]',
                'nitrogen': '[N]', 'азот': '[N]', 'n': '[N]',
                'oxygen': '[O]', 'кисень': '[O]', 'кислород': '[O]', 'o': '[O]',
                'sulfur': '[S]', 'сірка': '[S]', 'сера': '[S]', 's': '[S]',
                'chlorine': '[Cl]', 'хлор': '[Cl]', 'cl': '[Cl]',
                'bromine': '[Br]', 'бром': '[Br]', 'br': '[Br]',
            }
            raw_t = str(agent_decision.get('target', '')).lower().strip("[]'")
            if raw_t in element_map:
                agent_decision['target'] = element_map[raw_t]
                print(f"   ⚙️ [Middleware] SMARTS conversion: {agent_decision['target']}")

        # 3. Абсолютний захист SMILES (пріоритет над словами)
        smiles_candidates = [
            word for word in user_input.split() 
            if len(word) > 4 and any(c in word for c in '(=#123456789)') and 'c' in word.lower()
        ]
        if smiles_candidates and tool not in ['CLUSTER', 'DOCK']:
            exact_smiles = max(smiles_candidates, key=len).strip(".,;:!?")
            if exact_smiles != agent_decision.get('target'):
                agent_decision['target'] = exact_smiles
                print(f"   ⚙️ [Middleware] Recovered SMILES: {exact_smiles}")
        # =================================================================
        print(f"🤖 AI Reasoning ({ai_time:.2f}s): {agent_decision.get('reasoning')}")
        print(f"🛠  Tool: {agent_decision.get('tool')} | Target: {agent_decision.get('target')}")
        
        tool_cutoff = agent_decision.get('cutoff', 0.6)
        tool_method = agent_decision.get('method', 'morgan').lower()
        
        tool_start = time.time()
        tool_res = execute_tool(agent_decision.get('tool'), agent_decision.get('target'), tool_cutoff, tool_method)
        tool_time = time.time() - tool_start
        
        print(f"{tool_res}")
        print(f"⏱  Total time: {ai_time + tool_time:.2f}s (AI: {ai_time:.1f}s, Tool: {tool_time:.1f}s)")

    except Exception as e:
        print(f"[ERR] API Error: {e}")

    import gc
    gc.collect()