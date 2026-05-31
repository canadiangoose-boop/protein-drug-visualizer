# 🧬 Interactive Protein-Drug Binding Visualizer

A real-time 3D computational biophysics simulation built in Python.  
Visualize how a drug molecule docks into a protein binding pocket — with live force calculations and an **induced-fit conformational change animation**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyVista](https://img.shields.io/badge/PyVista-0.43%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **CPK 3D Rendering** | Atoms rendered as spheres with standard chemistry colours (grey=C, blue=N, red=O, yellow=S) |
| ⚗️ **4 Force Types** | Van der Waals · Electrostatic · Hydrogen Bonds · Hydrophobic — all calculated in real time |
| 🎛️ **Interactive Sliders** | Dock and rotate the drug molecule with live energy readout |
| 🔗 **Interaction Lines** | Colour-coded dashed lines show every active non-covalent bond |
| 🤝 **Induced Fit** | When binding energy crosses −7.5 kcal/mol, the protein clamps around the drug (smooth animation) |
| 🏗️ **Multi-Agent Architecture** | Physics · Rendering · Animation as independent agent classes |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/canadiangoose-boop/protein-drug-visualizer.git
cd protein-drug-visualizer

# 2. Install dependencies
pip install pyvista numpy

# 3. Run
python protein_visualizer.py
```

---

## 🎮 How to Use

1. **Drag "Approach Drug"** slider → moves the drug into the binding pocket
2. **Drag "Rotate Drug"** slider → rotates the drug to find the best fit
3. Watch **interaction lines** appear in real time:
   - 🩵 **Cyan** = Hydrogen bonds
   - 🟠 **Orange** = Ionic / electrostatic
   - 🟢 **Green** = Hydrophobic contacts
   - ⚫ **Grey** = Van der Waals
4. When **ΔG < −7.5 kcal/mol** → the protein automatically **clamps around the drug** (induced-fit)
5. **RESET** button → restore starting positions

---

## ⚛️ Physics Engine

### 1. Van der Waals — Lennard-Jones 12-6

$$E_{LJ} = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

- $(\sigma/r)^{12}$ → Pauli repulsion at short range
- $(\sigma/r)^6$ → London dispersion attraction
- Uses Lorentz-Berthelot combining rules for mixed atom pairs

### 2. Electrostatics — Coulomb

$$E_{elec} = \frac{k \cdot q_1 q_2}{\varepsilon(r) \cdot r}, \quad \varepsilon(r) = 4r$$

Distance-dependent dielectric approximates solvent screening inside the pocket.

### 3. Hydrogen Bonds

$$E_{HB} = D \cdot \left[5\left(\frac{r_0}{r}\right)^{12} - 6\left(\frac{r_0}{r}\right)^{10}\right]$$

- Optimal donor–acceptor distance $r_0 = 2.9$ Å
- Well depth $D = 4.0$ kcal/mol
- Only active for $2.3 \leq r \leq 3.5$ Å

### 4. Hydrophobic Effect

Contact-area scoring: −1.0 kcal/mol per non-polar pair at 3.5–5.0 Å contact distance. Models the entropy-driven burial of hydrophobic surfaces in water.

---

## 🏗️ Architecture — Multi-Agent Pattern

```
┌──────────────────────────────────────────────────────┐
│                  ORCHESTRATOR AGENT                  │
│            (ProteinVisualizerApp class)              │
│  Receives user input · Dispatches · Merges results   │
└──────────┬──────────────────┬──────────┬─────────────┘
           │                  │          │
┌──────────▼───┐  ┌───────────▼────┐  ┌─▼────────────┐
│ PHYSICS AGENT│  │   UI AGENT     │  │ ANIM. AGENT  │
│ PhysicsEngine│  │  MeshBuilder   │  │InducedFitAnim│
│              │  │                │  │              │
│ · LJ  (vdW)  │  │ · atom_sphere  │  │ · smooth_step│
│ · Coulomb    │  │ · bond_tube    │  │ · compute_    │
│ · H-bonds    │  │ · dashed_tube  │  │   target     │
│ · Hydrophob. │  │ · update_hud   │  │ · step() loop│
└──────────────┘  └────────────────┘  └──────────────┘
```

This mirrors Claude's **Research multi-agent architecture** — an orchestrator coordinates specialised subagents, each responsible for one domain.

---

## 🧪 Molecular System

### Protein — CDK2-like Kinase Binding Pocket
Inspired by **CDK2 (Cyclin-Dependent Kinase 2)**, a key cancer drug target.  
Key residues modelled: Leu83, Val56, Ile70, Met120, Gly121, Lys45, Asp166.

### Drug — ATP-competitive Inhibitor Scaffold
Inspired by clinical kinase inhibitors (imatinib / erlotinib scaffold).  
Features a 6-membered aromatic ring with two hinge-binding arms and an amide tail.

---

## 📁 File Structure

```
protein-drug-visualizer/
├── protein_visualizer.py   # Full application (single file)
├── requirements.txt        # pip dependencies
└── README.md
```

---

## 📋 Requirements

- Python 3.8+
- `pyvista >= 0.43.0`
- `numpy >= 1.24.0`

---

## 🔭 Future Ideas

- [ ] Load real PDB structures (HIV protease, COX-2, DHFR)
- [ ] Add atom editor (click to add/remove atoms from drug)
- [ ] Export binding energy plot as CSV
- [ ] Monte Carlo docking search
- [ ] PyMOL-style surface rendering

---

## 📚 References

- Lennard-Jones potential — *Proc. R. Soc. London*, 1924
- Induced-fit model — Koshland, D.E. (1958) *PNAS*
- Distance-dependent dielectric — Mehler & Solmajer (1991) *Protein Engineering*
- CDK2 structure — PDB: 1FIN

---

*Built with Python · PyVista · NumPy*
