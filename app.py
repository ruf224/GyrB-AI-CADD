
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="GyrB AI-CADD Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ================================================================
# LOAD RESOURCES
# ================================================================

@st.cache_resource
def load_lbdd_model():

    model_path = os.path.join(
        MODEL_DIR,
        "gyrb_lbdd_model.joblib"
    )

    return joblib.load(model_path)


@st.cache_resource
def load_reference_fingerprints():

    fp_path = os.path.join(
        MODEL_DIR,
        "gyrb_lbdd_reference_fingerprints.pkl"
    )

    with open(fp_path, "rb") as f:
        return joblib.load(f)


@st.cache_data
def load_datasets():

    integrated = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "GyrB_integrated_CADD_candidates.csv"
        )
    )

    admet = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "GyrB_de_novo_ADMET_candidates.csv"
        )
    )

    denovo = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "GyrB_de_novo_candidates.csv"
        )
    )

    validation = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "GyrB_LBDD_validation.csv"
        )
    )

    return integrated, admet, denovo, validation


@st.cache_data
def load_reference_ligand():

    path = os.path.join(
        DATA_DIR,
        "GyrB_SBDD_reference_ligand.csv"
    )

    return pd.read_csv(path)


# ================================================================
# INITIALIZE
# ================================================================

try:

    model = load_lbdd_model()
    reference_fps = load_reference_fingerprints()
    integrated_df, admet_df, denovo_df, validation_df = load_datasets()
    reference_df = load_reference_ligand()

except Exception as e:

    st.error("The application could not load the required project files.")

    st.code(str(e))

    st.stop()


# ================================================================
# MORGAN FINGERPRINT
# ================================================================

MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def get_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    fp = MORGAN_GENERATOR.GetFingerprint(mol)

    return mol, fp


# ================================================================
# APPLICABILITY DOMAIN
# ================================================================

def calculate_similarity_to_training(fp):

    similarities = []

    for ref_fp in reference_fps:

        try:

            similarity = DataStructs.TanimotoSimilarity(
                fp,
                ref_fp
            )

            similarities.append(similarity)

        except Exception:
            pass

    if not similarities:
        return np.nan, np.nan

    similarities = np.array(similarities)

    maximum = float(np.max(similarities))

    top_n = min(5, len(similarities))

    top_mean = float(
        np.mean(
            np.sort(similarities)[-top_n:]
        )
    )

    return maximum, top_mean


def assign_applicability_domain(max_similarity):

    if max_similarity >= 0.60:
        return "In-domain"

    elif max_similarity >= 0.40:
        return "Borderline"

    else:
        return "Out-of-domain"


# ================================================================
# DRUG-LIKENESS
# ================================================================

def calculate_properties(mol):

    return {

        "Molecular Weight":
            round(Descriptors.MolWt(mol), 2),

        "LogP":
            round(Crippen.MolLogP(mol), 2),

        "TPSA":
            round(rdMolDescriptors.CalcTPSA(mol), 2),

        "HBD":
            int(Lipinski.NumHDonors(mol)),

        "HBA":
            int(Lipinski.NumHAcceptors(mol)),

        "Rotatable Bonds":
            int(Lipinski.NumRotatableBonds(mol)),

        "Aromatic Rings":
            int(Lipinski.NumAromaticRings(mol)),

        "Heavy Atoms":
            int(mol.GetNumHeavyAtoms())
    }


def lipinski_status(props):

    violations = 0

    if props["Molecular Weight"] > 500:
        violations += 1

    if props["LogP"] > 5:
        violations += 1

    if props["HBD"] > 5:
        violations += 1

    if props["HBA"] > 10:
        violations += 1

    if violations == 0:
        return "Lipinski-compatible", violations

    elif violations == 1:
        return "One Lipinski violation", violations

    else:
        return "Multiple Lipinski violations", violations


# ================================================================
# SBDD STRUCTURE-INFORMED PROXY
# ================================================================

def calculate_sbdd_score(mol, fp):

    ref_smiles = reference_df.iloc[0]["Canonical_SMILES"]

    ref_mol = Chem.MolFromSmiles(ref_smiles)

    if ref_mol is None:
        return np.nan

    ref_fp = MORGAN_GENERATOR.GetFingerprint(ref_mol)

    fp_similarity = DataStructs.TanimotoSimilarity(
        fp,
        ref_fp
    )

    candidate_props = calculate_properties(mol)
    reference_props = calculate_properties(ref_mol)

    # Physicochemical similarity
    mw_similarity = max(
        0,
        1 - abs(
            candidate_props["Molecular Weight"]
            - reference_props["Molecular Weight"]
        ) / 300
    )

    logp_similarity = max(
        0,
        1 - abs(
            candidate_props["LogP"]
            - reference_props["LogP"]
        ) / 5
    )

    tpsa_similarity = max(
        0,
        1 - abs(
            candidate_props["TPSA"]
            - reference_props["TPSA"]
        ) / 100
    )

    physicochemical_similarity = np.mean([
        mw_similarity,
        logp_similarity,
        tpsa_similarity
    ])

    # Simple pharmacophore-like similarity
    donor_similarity = max(
        0,
        1 - abs(
            candidate_props["HBD"]
            - reference_props["HBD"]
        ) / 5
    )

    acceptor_similarity = max(
        0,
        1 - abs(
            candidate_props["HBA"]
            - reference_props["HBA"]
        ) / 8
    )

    aromatic_similarity = max(
        0,
        1 - abs(
            candidate_props["Aromatic Rings"]
            - reference_props["Aromatic Rings"]
        ) / 5
    )

    pharmacophore_similarity = np.mean([
        donor_similarity,
        acceptor_similarity,
        aromatic_similarity
    ])

    score = (

        0.50 * fp_similarity
        +
        0.30 * pharmacophore_similarity
        +
        0.20 * physicochemical_similarity

    )

    return float(score)


# ================================================================
# HEADER
# ================================================================

st.title("🧬 GyrB AI-CADD Platform")

st.markdown(
    """
### AI-assisted computational drug discovery for DNA gyrase B

This platform integrates **ligand-based machine learning, 
structure-informed scoring, constrained de-novo design,
drug-likeness/ADMET profiling, applicability-domain analysis,
and integrated computational prioritization**.
"""
)

st.info(
    "Target: DNA gyrase B (GyrB) | "
    "Structural reference: PDB 4DUH"
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("Platform Navigation")

page = st.sidebar.radio(
    "Select module",
    [
        "🏠 Overview",
        "🧬 Single Compound Analysis",
        "🧩 Structure-Informed SBDD",
        "✨ De-novo Candidates",
        "🧪 ADMET / Drug-Likeness",
        "🏆 Integrated Ranking",
        "📊 Model Validation",
        "ℹ️ Methodology & Limitations"
    ]
)


# ================================================================
# OVERVIEW
# ================================================================

if page == "🏠 Overview":

    st.header("Platform Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Training structures",
        "653"
    )

    col2.metric(
        "Generated candidates",
        "514"
    )

    col3.metric(
        "High-priority candidates",
        "120"
    )

    col4.metric(
        "GyrB structure",
        "4DUH"
    )

    st.subheader("Workflow")

    st.markdown(
        """
        **1. Input compound**

        ↓

        **2. LBDD activity prediction**

        ↓

        **3. Applicability-domain assessment**

        ↓

        **4. Structure-informed SBDD scoring**

        ↓

        **5. ADMET / drug-likeness profiling**

        ↓

        **6. Integrated CADD prioritization**

        ↓

        **7. Candidate selection for experimental follow-up**
        """
    )

    st.subheader("Integrated scoring framework")

    st.markdown(
        """
        | Component | Weight |
        |---|---:|
        | LBDD activity score | 40% |
        | Structure-informed SBDD score | 30% |
        | ADMET score | 20% |
        | Applicability-domain score | 10% |
        """
    )

    st.warning(
        "The integrated score is a computational prioritization "
        "framework and is not an experimentally validated binding "
        "or potency prediction."
    )


# ================================================================
# SINGLE COMPOUND ANALYSIS
# ================================================================

elif page == "🧬 Single Compound Analysis":

    st.header("Single Compound Analysis")

    st.write(
        "Enter a SMILES string to evaluate a compound using the "
        "trained GyrB LBDD model and structure-informed workflow."
    )

    smiles = st.text_area(
        "Compound SMILES",
        value="",
        height=120,
        placeholder="Example: CCO"
    )

    if st.button(
        "🔬 Analyze Compound",
        type="primary"
    ):

        if not smiles.strip():

            st.warning(
                "Please enter a SMILES string."
            )

        else:

            mol, fp = get_fingerprint(
                smiles.strip()
            )

            if mol is None:

                st.error(
                    "Invalid SMILES. Please check the structure."
                )

            else:

                canonical = Chem.MolToSmiles(mol)

                probability = float(
                    model.predict_proba(
                        [np.array(fp)]
                    )[0][1]
                )

                predicted_class = int(
                    model.predict(
                        [np.array(fp)]
                    )[0]
                )

                max_similarity, top5_similarity = (
                    calculate_similarity_to_training(fp)
                )

                ad_domain = assign_applicability_domain(
                    max_similarity
                )

                properties = calculate_properties(mol)

                lipinski, violations = lipinski_status(
                    properties
                )

                sbdd_score = calculate_sbdd_score(
                    mol,
                    fp
                )

                st.success(
                    "Compound successfully analyzed."
                )

                st.subheader("LBDD Prediction")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Active probability",
                    f"{probability:.3f}"
                )

                c2.metric(
                    "Prediction",
                    "Active-like"
                    if predicted_class == 1
                    else "Inactive-like"
                )

                c3.metric(
                    "Applicability domain",
                    ad_domain
                )

                st.subheader(
                    "Structure-informed SBDD"
                )

                st.metric(
                    "SBDD proxy score",
                    f"{sbdd_score:.3f}"
                )

                st.subheader(
                    "Chemical Properties"
                )

                property_df = pd.DataFrame(
                    {
                        "Property":
                            list(properties.keys()),

                        "Value":
                            list(properties.values())
                    }
                )

                st.dataframe(
                    property_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader(
                    "Chemical-space similarity"
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "Maximum training Tanimoto",
                    f"{max_similarity:.3f}"
                )

                c2.metric(
                    "Mean top-5 Tanimoto",
                    f"{top5_similarity:.3f}"
                )

                st.subheader(
                    "Drug-likeness"
                )

                st.write(
                    f"**{lipinski}**"
                )

                st.write(
                    f"Lipinski violations: {violations}"
                )

                st.caption(
                    f"Canonical SMILES: {canonical}"
                )


# ================================================================
# SBDD
# ================================================================

elif page == "🧩 Structure-Informed SBDD":

    st.header(
        "Structure-Informed SBDD"
    )

    st.write(
        "This module compares compounds against the "
        "GyrB reference ligand from PDB 4DUH."
    )

    ref = reference_df.iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "PDB",
        str(ref["PDB_ID"])
    )

    c2.metric(
        "Molecular Weight",
        f"{ref['Molecular_Weight']:.2f}"
    )

    c3.metric(
        "TPSA",
        f"{ref['TPSA']:.2f}"
    )

    st.subheader(
        "Reference ligand"
    )

    st.code(
        ref["Canonical_SMILES"]
    )

    st.markdown(
        """
        **Structure-informed score components**

        - 50% molecular fingerprint similarity
        - 30% pharmacophore-like similarity
        - 20% physicochemical similarity
        """
    )

    st.warning(
        "This is a structure-informed SBDD proxy. "
        "It is not molecular docking and does not calculate "
        "binding free energy."
    )


# ================================================================
# DE-NOVO CANDIDATES
# ================================================================

elif page == "✨ De-novo Candidates":

    st.header(
        "GyrB De-novo Candidate Library"
    )

    st.write(
        "Constrained aromatic substitutions were used to generate "
        "new analogues within the learned chemical space."
    )

    display_cols = [
        "Candidate_SMILES",
        "Parent_SMILES",
        "Generation_Method",
        "MW",
        "LogP",
        "TPSA",
        "LBDD_Activity_Probability",
        "Max_Training_Tanimoto",
        "Applicability_Domain"
    ]

    available_cols = [
        c for c in display_cols
        if c in denovo_df.columns
    ]

    top_n = st.slider(
        "Number of candidates",
        min_value=10,
        max_value=min(100, len(denovo_df)),
        value=20
    )

    sort_col = "LBDD_Activity_Probability"

    table = denovo_df.sort_values(
        sort_col,
        ascending=False
    ).head(top_n)

    st.dataframe(
        table[available_cols],
        use_container_width=True,
        hide_index=True
    )

    csv_data = table.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download displayed candidates",
        data=csv_data,
        file_name="GyrB_denovo_candidates.csv",
        mime="text/csv"
    )


# ================================================================
# ADMET
# ================================================================

elif page == "🧪 ADMET / Drug-Likeness":

    st.header(
        "ADMET / Drug-Likeness Profiling"
    )

    st.write(
        "Descriptor-based computational risk profiling of generated "
        "GyrB candidates."
    )

    if "ADMET_Risk_Category" in admet_df.columns:

        risk_counts = (
            admet_df["ADMET_Risk_Category"]
            .value_counts()
        )

        st.subheader(
            "ADMET risk distribution"
        )

        st.bar_chart(
            risk_counts
        )

    if "Drug_Likeness" in admet_df.columns:

        drug_counts = (
            admet_df["Drug_Likeness"]
            .value_counts()
        )

        st.subheader(
            "Drug-likeness distribution"
        )

        st.bar_chart(
            drug_counts
        )

    st.subheader(
        "Top candidates after ADMET profiling"
    )

    cols = [
        "Candidate_SMILES",
        "LBDD_Activity_Probability",
        "ADMET_Score",
        "ADMET_Risk_Category",
        "Drug_Likeness",
        "MW",
        "LogP",
        "TPSA"
    ]

    cols = [
        c for c in cols
        if c in admet_df.columns
    ]

    st.dataframe(
        admet_df.sort_values(
            "LBDD_Activity_Probability",
            ascending=False
        ).head(25)[cols],
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "These ADMET results are descriptor-based computational "
        "risk flags. They are not experimental CYP, hERG, "
        "P-gp, hepatotoxicity, genotoxicity, or clinical ADMET "
        "predictions."
    )


# ================================================================
# INTEGRATED RANKING
# ================================================================

elif page == "🏆 Integrated Ranking":

    st.header(
        "Integrated GyrB CADD Candidate Ranking"
    )

    st.write(
        "Candidates are prioritized using the integrated "
        "computational scoring framework."
    )

    priority_filter = st.multiselect(
        "Computational priority",
        options=sorted(
            integrated_df[
                "Computational_Priority"
            ].dropna().unique()
        ),
        default=sorted(
            integrated_df[
                "Computational_Priority"
            ].dropna().unique()
        )
    )

    filtered = integrated_df[
        integrated_df[
            "Computational_Priority"
        ].isin(priority_filter)
    ].copy()

    st.metric(
        "Candidates displayed",
        len(filtered)
    )

    columns = [
        "Integrated_Rank",
        "Candidate_SMILES",
        "Generation_Method",
        "LBDD_Score",
        "SBDD_Structure_Informed_Score",
        "ADMET_Integrated_Score",
        "Max_Training_Tanimoto",
        "Applicability_Domain",
        "ADMET_Risk_Category",
        "Drug_Likeness",
        "Integrated_CADD_Score",
        "Computational_Priority"
    ]

    columns = [
        c for c in columns
        if c in filtered.columns
    ]

    st.dataframe(
        filtered.sort_values(
            "Integrated_Rank"
        )[columns].head(100),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download integrated candidates",
        data=csv_data,
        file_name="GyrB_integrated_CADD_candidates.csv",
        mime="text/csv"
    )

    st.warning(
        "Integrated ranking represents computational prioritization "
        "and should not be interpreted as experimentally confirmed "
        "GyrB potency or binding."
    )


# ================================================================
# MODEL VALIDATION
# ================================================================

elif page == "📊 Model Validation":

    st.header(
        "GyrB LBDD Model Validation"
    )

    st.subheader(
        "Scaffold-aware validation"
    )

    metrics = {
        "Accuracy": 0.8246,
        "Balanced Accuracy": 0.7637,
        "ROC-AUC": 0.9175,
        "PR-AUC": 0.9708,
        "F1 Score": 0.8810,
        "MCC": 0.5506,
        "Precision": 0.8605,
        "Sensitivity": 0.9024,
        "Specificity": 0.6250
    }

    metric_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        }
    )

    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Validation interpretation"
    )

    st.write(
        """
        The model was evaluated using a scaffold-aware split to
        reduce direct chemical-series leakage between training and
        validation compounds.

        The validation results support the use of the model as a
        computational prioritization component. They do not establish
        experimental activity for newly generated molecules.
        """
    )


# ================================================================
# METHODOLOGY
# ================================================================

elif page == "ℹ️ Methodology & Limitations":

    st.header(
        "Methodology & Limitations"
    )

    st.subheader(
        "Target"
    )

    st.write(
        "DNA gyrase B (GyrB), using PDB 4DUH as the structural "
        "reference."
    )

    st.subheader(
        "LBDD"
    )

    st.write(
        """
        A Random Forest classifier was trained using Morgan
        fingerprints generated from a curated GyrB IC50 dataset.
        A scaffold-aware validation split was used.
        """
    )

    st.subheader(
        "Structure-informed SBDD"
    )

    st.write(
        """
        Candidate molecules are compared with the reference ligand
        using molecular fingerprint similarity, simple
        pharmacophore-like feature similarity, and physicochemical
        similarity.
        """
    )

    st.subheader(
        "De-novo design"
    )

    st.write(
        """
        Candidate molecules were generated through constrained
        aromatic substitutions around active-like starting
        scaffolds.
        """
    )

    st.subheader(
        "ADMET"
    )

    st.write(
        """
        Current ADMET assessment is descriptor-based. It is intended
        for early computational triage rather than replacement of
        experimental ADMET studies.
        """
    )

    st.subheader(
        "Important limitations"
    )

    limitations = [
        "The SBDD module is not molecular docking.",
        "No binding free energy is calculated.",
        "Generated molecules have not been experimentally validated.",
        "The LBDD model predicts an activity class rather than a validated experimental IC50.",
        "Applicability-domain thresholds are heuristic.",
        "Integrated weights are heuristic and not experimentally optimized.",
        "ADMET risk flags are computational descriptors rather than clinical predictions.",
        "Experimental biochemical and microbiological validation remains necessary."
    ]

    for item in limitations:
        st.write(
            "• " + item
        )

    st.info(
        "This platform is intended for computational research and "
        "hypothesis generation."
    )


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.caption(
    "GyrB AI-CADD Platform | "
    "AI-assisted computational drug discovery prototype"
)
