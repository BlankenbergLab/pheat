# Quick Start

## Install

```bash
python -m pip install -e .
```

Optional capabilities are grouped into extras. For example, the web app can be
installed with:

```bash
python -m pip install -e ".[web]"
```

The full development feature surface is available with:

```bash
python -m pip install -e ".[all]"
```

## Load A Structure

```python
from pheat import load_pdb

structure = load_pdb("tests/fixtures/2mu7.pdb")
print(structure.name, len(structure.atoms))
```

## Convert To Residue Geometry

```python
from pheat import load_pdb, structure_to_residue_geometry

structure = load_pdb("tests/fixtures/2mu7.pdb")
geometry = structure_to_residue_geometry(
    structure,
    stored_angles="all",
    angle_units="radians",
)
print(geometry["format"], len(geometry["residues"]))
```

## Reconstruct Heavy Atoms

```python
from pheat import structure_from_residue_geometry

reconstructed = structure_from_residue_geometry(geometry)
print(len(reconstructed.atoms))
```

## Score A Structure

```python
from pheat import load_pdb, score_structure

structure = load_pdb("tests/fixtures/2mu7.pdb")
score = score_structure(structure, model="generic")
print(score.model, score.total, score.units)
```

## Command Line

```bash
pheat pdb-to-geometry tests/fixtures/2mu7.pdb -o 2mu7.geometry.json
pheat geometry-to-structure 2mu7.geometry.json -o 2mu7.reconstructed.json
pheat score tests/fixtures/2mu7.pdb --model generic
```

For more commands:

```bash
pheat --help
```
