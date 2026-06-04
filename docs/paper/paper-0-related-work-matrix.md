# Related Work Matrix

| Tool or dataset | Relationship to PHEAT |
| --- | --- |
| Biopython Bio.PDB | General parser; PHEAT complements it with corpus manifests and provenance. |
| Gemmi | High-fidelity crystallographic/mmCIF library; PHEAT complements it with user-defined corpus outputs. |
| MDAnalysis | Analysis framework; PHEAT complements it with rebuildable reference datasets. |
| MDTraj | Trajectory/structure analysis toolkit; PHEAT complements it with corpus generation and manifest checksums. |
| Biopython internal_coords | Internal-coordinate implementation; PHEAT needs careful contract-matched reconstruction comparison. |
| PeptideBuilder | Angle-to-peptide builder; PHEAT complements rather than replaces it. |
| PULCHRA | Reduced-representation reconstruction; paper-0 compares PULCHRA in reduced-input and PHEAT artifact-assisted reconstruction tasks. |
| SidechainNet | Fixed ML-oriented dataset; PHEAT adds user-defined corpus generation and provenance. |
| PISCES | Sequence-culling server; PHEAT can record or consume comparable filters but does not replace PISCES. |
| OpenMM | Molecular simulation/energy engine; PHEAT can record optional external backend status. |
| GROMACS | Molecular simulation engine; PHEAT can delegate optional scoring but does not reimplement it. |
| AmberTools | Molecular modeling tools; PHEAT can record optional scoring provenance. |
| FreeSASA | SASA calculator; PHEAT can use it as optional backend/provenance input. |
