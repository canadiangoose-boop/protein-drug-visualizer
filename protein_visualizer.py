#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║       INTERACTIVE PROTEIN-DRUG BINDING VISUALIZER  v1.0                ║
║       Computational Biophysics Simulation  |  Python + PyVista          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  ARCHITECTURE   Multi-Agent Orchestrator Pattern (see §8)               ║
║  PHYSICS        LJ · Coulomb · H-Bond · Hydrophobic engines (see §5)   ║
║  RENDERING      3D CPK coloring, real-time force lines (see §6)        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Install :  pip install pyvista numpy                                   ║
║  Run     :  python protein_visualizer.py                                ║
╚══════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MULTI-AGENT ARCHITECTURE  (§8 implements this pattern)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌──────────────────────────────────────────────────────────────┐
    │                    ORCHESTRATOR AGENT                        │
    │               (ProteinVisualizerApp class)                   │
    │   Receives user input · Dispatches tasks · Merges results    │
    └────────────┬──────────────────┬──────────────┬──────────────┘
                 │                  │              │
    ┌────────────▼───┐   ┌──────────▼────┐  ┌─────▼────────────┐
    │  PHYSICS AGENT │   │   UI AGENT    │  │ ANIMATION AGENT  │
    │  PhysicsEngine │   │  MeshBuilder  │  │ InducedFitAnim.  │
    │                │   │               │  │                  │
    │ · lennard_jones│   │ · atom_sphere │  │ · smooth_step    │
    │ · coulomb      │   │ · bond_tube   │  │ · compute_target │
    │ · hydrogen_bond│   │ · interaction │  │ · step() loop    │
    │ · hydrophobic  │   │   _tube       │  │ · threading cb   │
    │ · score()      │   │ · update_hud  │  │                  │
    └────────────────┘   └───────────────┘  └──────────────────┘

 Mapping to Claude's Research multi-agent pattern:
   Orchestrator    = Lead Research Agent (plans, coordinates)
   PhysicsEngine   = Subagent #1 — force calculation specialist
   MeshBuilder     = Subagent #2 — rendering / state specialist
   InducedFitAnim  = Subagent #3 — animation sequencing specialist

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ═══════════════════════════════════════════════════════════════════════════
# §1  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import sys
import time
import threading
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista not found.\n  Install with:  pip install pyvista")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# §2  CONSTANTS & LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════

# ── Rendering ───────────────────────────────────────────────────────────
ATOM_SCALE        = 0.38   # fraction of VDW radius shown (keeps pocket open)
BOND_RADIUS       = 0.07   # Å — covalent bond tube radius
IACT_RADIUS       = 0.03   # Å — non-covalent interaction line radius
APPROACH_DISTANCE = 12.0   # Å — distance drug travels from start → docked

# ── Physics thresholds ──────────────────────────────────────────────────
INDUCED_FIT_THRESHOLD = -7.5   # kcal/mol — binding energy that triggers clamp
MAX_POCKET_SHIFT      = 0.50   # Å — maximum protein atom displacement
CUTOFF                = 12.0   # Å — ignore pairwise interactions beyond this

# ── CPK element colours (standard chemistry colouring scheme) ───────────
CPK = {
    'C': '#909090',   # carbon   → grey
    'N': '#4169E1',   # nitrogen → royal blue
    'O': '#FF2200',   # oxygen   → red
    'H': '#F5F5F5',   # hydrogen → near-white
    'S': '#FFD700',   # sulphur  → gold
    'P': '#FF8C00',   # phosphorus → dark orange
}

# ── Non-covalent interaction line colours ───────────────────────────────
CLR_HBOND = '#00FFFF'   # cyan   — hydrogen bonds
CLR_IONIC = '#FF7F00'   # orange — ionic / electrostatic
CLR_HYDRO = '#88FF88'   # green  — hydrophobic
CLR_VDW   = '#555555'   # grey   — van der Waals

# ── Van der Waals radii (Å) ─────────────────────────────────────────────
VDW = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'H': 1.20, 'S': 1.80, 'P': 1.80}

# ── Lennard-Jones parameters  ε (kcal/mol) and σ (Å) per element ────────
LJ_PARAMS = {
    'C': (0.086, 3.50), 'N': (0.170, 3.25), 'O': (0.210, 2.96),
    'H': (0.030, 2.42), 'S': (0.250, 3.56), 'P': (0.200, 3.74),
}


# ═══════════════════════════════════════════════════════════════════════════
# §3  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class Atom:
    """
    Represents a single atom with all properties needed for
    biophysical force calculations and 3D rendering.

    Parameters
    ----------
    id                Unique string  (e.g. 'P07', 'D03')
    element           Chemical symbol in {C, N, O, H, S, P}
    pos               [x, y, z] coordinates in Ångstroms
    charge            Partial atomic charge in elementary charges (e)
    hydrophobic       True → non-polar atom — contributes to hydrophobic effect
    donor             True → H-bond donor (NH or OH group)
    acceptor          True → H-bond acceptor (lone-pair on O or N)
    label             Display label (residue name for protein atoms)
    """
    __slots__ = ('id', 'element', 'position', 'charge',
                 'is_hydrophobic', 'is_hbond_donor', 'is_hbond_acceptor',
                 'label', '_origin')

    def __init__(self, id, element, pos, charge,
                 hydrophobic, donor, acceptor, label=''):
        self.id                = id
        self.element           = element
        self.position          = np.array(pos, dtype=float)
        self.charge            = float(charge)
        self.is_hydrophobic    = hydrophobic
        self.is_hbond_donor    = donor
        self.is_hbond_acceptor = acceptor
        self.label             = label
        self._origin           = self.position.copy()

    def reset(self):
        """Restore original coordinates (used on RESET button)."""
        self.position = self._origin.copy()

    @property
    def color(self):
        return CPK.get(self.element, '#FF00FF')

    @property
    def radius(self):
        return VDW.get(self.element, 1.50) * ATOM_SCALE

    def __repr__(self):
        return f"Atom({self.id} [{self.element}] @ {self.position.round(2)})"


# ═══════════════════════════════════════════════════════════════════════════
# §4  MOLECULAR DATA
# ═══════════════════════════════════════════════════════════════════════════
#
#  Protein: simplified CDK2-like kinase active site
#           Key residues: Leu/Val/Ile hydrophobic ceiling,
#           Met/Gly backbone hinge (H-bond donors/acceptors),
#           Lys45 (cation) and Asp166 (anion) ionic pair.
#
#  Drug:    ATP-competitive inhibitor scaffold
#           Purine-like ring with amide chain — designed to
#           complement the hinge H-bond donors/acceptors.
#
#  Atom() call signature:
#    id, element, [x,y,z], charge, hydrophobic, hbond_donor, hbond_acceptor, label

def make_protein_atoms():
    """
    Returns a cup-shaped binding pocket opening upward.

    Layout (docked drug sits at center, Y ≈ 0):
      Hydrophobic ceiling  Y = +2.5 → +3.5   (Leu/Val/Ile/Cys cluster)
      Hinge backbone       Y =  0,  X = ±5.5  (H-bond donors + acceptors)
      Ionic anchors        Y = -3.2, Z = ±3.0  (Lys+ and Asp-)
      Hydrophobic floor    Y = -6.0             (Ile/Phe aliphatic cores)

    All protein atoms are ≥ 3.0 Å from every docked drug atom (verified).
    """
    return [
        # ── Hydrophobic ceiling ──────────────────────────────────────────
        Atom('P01','C', [-2.5,  3.5,  1.0], -0.10, True,  False, False, 'LEU83-CD2'),
        Atom('P02','C', [ 0.0,  3.5,  2.0], -0.10, True,  False, False, 'VAL56-CG1'),
        Atom('P03','C', [ 2.5,  3.5,  1.0], -0.10, True,  False, False, 'ILE70-CD1'),
        Atom('P04','C', [ 0.0,  3.5, -1.5], -0.10, True,  False, False, 'LEU102-CD'),
        Atom('P05','S', [ 0.0,  3.0,  3.5],  0.00, True,  False, False, 'CYS99-SG'),

        # ── Hinge backbone H-bond acceptors (backbone C=O) ──────────────
        Atom('P06','O', [-5.5,  0.0,  0.5], -0.60, False, False, True,  'MET120-O'),
        Atom('P07','O', [ 5.5,  0.0, -0.5], -0.60, False, False, True,  'GLY121-O'),

        # ── Hinge backbone H-bond donors (backbone N-H) ─────────────────
        Atom('P08','N', [-5.5, -1.0, -0.5], -0.40, False, True,  False, 'MET120-N'),
        Atom('P09','N', [ 5.5, -1.0,  0.5], -0.40, False, True,  False, 'GLY121-N'),

        # ── Ionic anchors ────────────────────────────────────────────────
        Atom('P10','N', [ 0.5, -3.2, -3.0],  0.80, False, True,  False, 'LYS45-NZ'),
        Atom('P11','O', [-0.5, -3.2,  3.0], -0.80, False, False, True,  'ASP166-OD'),

        # ── Hydrophobic floor ─────────────────────────────────────────────
        Atom('P12','C', [-1.5, -6.0,  1.0], -0.10, True,  False, False, 'ILE12-CD'),
        Atom('P13','C', [ 1.5, -6.0,  1.0], -0.10, True,  False, False, 'PHE80-CD'),
    ]


PROTEIN_BONDS = [
    ('P06','P08'), ('P07','P09'),
    ('P01','P02'), ('P02','P03'), ('P03','P04'),
    ('P12','P13'),
]


# Y offset that places the drug 12 Å above docked position
_Y_START = 12.0


def make_drug_atoms():
    """
    Returns the drug molecule at its STARTING position (12 Å above pocket).
    APPROACH_DISTANCE = 12.0 slides it down to the docked position at Y = 0.

    Docked geometry (Y values shown without _Y_START offset):
      6-membered ring centred at origin, Y ≈ 0 — slots between hinge and ceiling
      N7 arm extends to X = -2.5 (H-bond with hinge P08 N-H donor)
      O8 arm extends to X = +2.5 (H-bond with hinge P09 N-H donor)
      Amide tail descends to Y ≈ -3.2 (ionic contact with P10/P11)

    All docked drug–protein distances verified ≥ 3.0 Å (no atom clash).
    """
    Y = _Y_START
    return [
        # ── 6-membered aromatic ring ─────────────────────────────────────
        Atom('D01','N', [-1.2,  0.0 + Y,  0.5], -0.40, False, True,  False, 'N1'),
        Atom('D02','C', [-0.5,  0.5 + Y,  1.2], -0.10, True,  False, False, 'C2'),
        Atom('D03','C', [ 0.5,  0.5 + Y,  1.2],  0.10, True,  False, False, 'C3'),
        Atom('D04','C', [ 1.2,  0.0 + Y,  0.5], -0.10, True,  False, False, 'C4'),
        Atom('D05','C', [ 0.8,  0.0 + Y, -0.6], -0.10, True,  False, False, 'C5'),
        Atom('D06','C', [-0.8,  0.0 + Y, -0.6], -0.10, True,  False, False, 'C6'),

        # ── Hinge-binding arms (H-bond acceptors) ────────────────────────
        #   D07 N at X=-2.5 → 3.32 Å from P08 N (H-bond donor) ✓
        #   D08 O at X=+2.5 → 3.16 Å from P09 N (H-bond donor) ✓
        Atom('D07','N', [-2.5,  0.0 + Y,  0.5], -0.45, False, False, True,  'N7-accep'),
        Atom('D08','O', [ 2.5,  0.0 + Y,  0.5], -0.60, False, False, True,  'O8-accep'),

        # ── Amide tail (H-bond donor + acceptor for ionic region) ────────
        #   D09 N → 3.46 Å from P11 O (H-bond acceptor) ✓
        #   D10 O ← 3.00 Å from P10 N (H-bond donor)    ✓
        Atom('D09','N', [ 0.5, -1.8 + Y,  0.0], -0.50, False, True,  False, 'N9-amide'),
        Atom('D10','O', [ 0.5, -3.2 + Y,  0.0], -0.60, False, False, True,  'O10-amide'),
    ]


DRUG_BONDS = [
    ('D01','D02'), ('D02','D03'), ('D03','D04'), ('D04','D05'),
    ('D05','D06'), ('D06','D01'),   # 6-membered ring closure
    ('D01','D07'),                   # N7 hinge-binding arm
    ('D04','D08'),                   # O8 hinge-binding arm
    ('D04','D09'), ('D09','D10'),    # amide tail
]


# ═══════════════════════════════════════════════════════════════════════════
# §5  PHYSICS ENGINE  —  Agent 1  (Force Calculations)
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsEngine:
    """
    Calculates all non-covalent intermolecular interactions.

    All energies are returned in  kcal/mol.
    All distances are in  Ångstroms (Å).

    Four interaction types are modelled:
      1. Van der Waals   — Lennard-Jones 12-6 potential
      2. Electrostatics  — Coulomb's law, distance-dependent dielectric
      3. Hydrogen Bonds  — distance + geometry scoring function
      4. Hydrophobic     — contact-area based scoring
    """

    K_ELEC = 332.0   # kcal·Å / (mol·e²) — Coulomb constant in mol. units

    # ── Geometry helpers ─────────────────────────────────────────────────

    @staticmethod
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def angle_deg(va: np.ndarray, vb: np.ndarray, vc: np.ndarray) -> float:
        """Angle at vertex vb in the triplet va-vb-vc (degrees)."""
        ba = va - vb
        bc = vc - vb
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom < 1e-9:
            return 0.0
        return float(np.degrees(np.arccos(
            np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))

    # ── 1. Van der Waals — Lennard-Jones 12-6 ────────────────────────────

    def lennard_jones(self, r: float, elem1: str, elem2: str) -> float:
        """
        E_LJ = 4ε · [(σ/r)¹² − (σ/r)⁶]

        Combining rules (Lorentz-Berthelot):
          σ₁₂ = (σ₁ + σ₂) / 2       ← arithmetic mean
          ε₁₂ = √(ε₁ · ε₂)          ← geometric mean

        Physics:
          (σ/r)¹²  →  Pauli (electron-cloud) repulsion at short range
          (σ/r)⁶   →  London dispersion attraction at medium range
          Minimum (equilibrium) at r_min = 2^(1/6) · σ ≈ 1.12σ
        """
        if r < 0.8:
            return 500.0  # hard-core wall — prevents atom collapse

        e1, s1 = LJ_PARAMS.get(elem1, (0.10, 3.50))
        e2, s2 = LJ_PARAMS.get(elem2, (0.10, 3.50))

        eps = np.sqrt(e1 * e2)    # geometric mean for well depth
        sig = (s1 + s2) / 2.0    # arithmetic mean for radius

        sr = sig / r
        return 4.0 * eps * (sr**12 - sr**6)

    # ── 2. Electrostatics — Coulomb ───────────────────────────────────────

    def coulomb(self, r: float, q1: float, q2: float) -> float:
        """
        E_elec = k · q1 · q2 / [ε(r) · r]

        Uses a linear distance-dependent dielectric  ε(r) = 4r
        to approximate solvent screening in a binding pocket.
        This is standard in empirical scoring functions (e.g. DOCK, AutoDock).

        Physics:
          q1·q2 > 0  →  repulsion   (like-sign charges)
          q1·q2 < 0  →  attraction  (opposite-sign — ionic bond)
        """
        if r < 0.5:
            return np.sign(q1 * q2) * 500.0
        dielectric = 4.0 * r       # linear distance-dependent screening
        return self.K_ELEC * q1 * q2 / (dielectric * r)

    # ── 3. Hydrogen Bonds ─────────────────────────────────────────────────

    def hydrogen_bond(self, donor_pos: np.ndarray,
                      acceptor_pos: np.ndarray) -> float:
        """
        Simplified H-bond scoring using heavy-atom distance only
        (no explicit H position required).

        Full 3-body term:  E = D · cos²(θ_DHA) · [5(r₀/r)¹² − 6(r₀/r)¹⁰]
        This implementation applies the radial term with r = r_DA (donor–acceptor)
        and ideal distance  r₀ = 2.9 Å  (typical N···O H-bond).

        Physics:
          Ideal D–H···A angle: 180° (collinear) — strictly directional
          Optimal r_DA: 2.7–3.2 Å (H···A: 1.8–2.0 Å + D–H bond: ~1.0 Å)
          Well depth D: −3 to −8 kcal/mol  (weaker than covalent, ~5–40× kT)
          Angle < 120° → contribution drops to 0
        """
        r = self.dist(donor_pos, acceptor_pos)
        if r < 2.3 or r > 3.5:
            return 0.0

        r0 = 2.9    # Å — optimal donor···acceptor distance
        D  = 4.0    # kcal/mol — well depth magnitude (positive)
        # At r=r0: (5·1^12 - 6·1^10) = -1  → E = D * (-1) = -4.0 kcal/mol ✓

        ratio = r0 / r
        return D * (5.0 * ratio**12 - 6.0 * ratio**10)

    # ── 4. Hydrophobic Effect ─────────────────────────────────────────────

    def hydrophobic(self, r: float, atom1: Atom, atom2: Atom) -> float:
        """
        Hydrophobic (solvophobic) interaction score.

        Two non-polar atoms in contact in water → favourable ΔG because
        burying non-polar surfaces together releases ordered water molecules
        (entropy-driven, ΔG = ΔH − T·ΔS, dominant T·ΔS term).

        Score (simplified buried-contact model):
          r < 3.0 Å  →  mild clash penalty (+0.3 kcal/mol)
          3.0–5.0 Å  →  optimal contact   (−1.0 kcal/mol)
          5.0–6.5 Å  →  linear decay to 0
          r > 6.5 Å  →  no contribution

        Real implementations scale with buried solvent-accessible surface area.
        """
        if not (atom1.is_hydrophobic and atom2.is_hydrophobic):
            return 0.0
        if r < 3.0:
            return +0.3   # non-polar clash
        elif r < 5.0:
            return -1.0   # optimal hydrophobic burial
        elif r < 6.5:
            return -1.0 * (6.5 - r) / 1.5   # linear fade-out
        return 0.0

    # ── Master scoring function ───────────────────────────────────────────

    def score(self, drug_atoms: list, protein_atoms: list) -> dict:
        """
        Evaluates all pairwise interactions between the drug molecule
        and the protein binding pocket.

        Returns
        -------
        dict
          'vdw'          : float — total LJ van der Waals (kcal/mol)
          'electrostatic': float — total Coulomb (kcal/mol)
          'hbonds'       : float — total H-bond energy (kcal/mol)
          'hydrophobic'  : float — total hydrophobic score (kcal/mol)
          'total'        : float — sum of all four terms
          'n_hbonds'     : int   — count of detected H-bond contacts
          'interactions' : list  — (type, drug_id, prot_id, energy, midpoint)
        """
        out = {
            'vdw': 0.0, 'electrostatic': 0.0,
            'hbonds': 0.0, 'hydrophobic': 0.0,
            'total': 0.0, 'n_hbonds': 0,
            'interactions': []
        }

        for da in drug_atoms:
            for pa in protein_atoms:
                r = self.dist(da.position, pa.position)
                if r > CUTOFF:
                    continue
                mid = (da.position + pa.position) * 0.5

                # ── Van der Waals ────────────────────────────────────────
                lj = self.lennard_jones(r, da.element, pa.element)
                out['vdw'] += lj
                if abs(lj) > 0.15:
                    out['interactions'].append(
                        ('vdw', da.id, pa.id, lj, mid))

                # ── Electrostatics ───────────────────────────────────────
                if abs(da.charge) > 0.05 and abs(pa.charge) > 0.05:
                    el = self.coulomb(r, da.charge, pa.charge)
                    out['electrostatic'] += el
                    if abs(el) > 0.25:
                        out['interactions'].append(
                            ('ionic', da.id, pa.id, el, mid))

                # ── H-bonds: drug donor → protein acceptor ───────────────
                if da.is_hbond_donor and pa.is_hbond_acceptor:
                    hb = self.hydrogen_bond(da.position, pa.position)
                    out['hbonds'] += hb
                    if hb < -0.5:
                        out['n_hbonds'] += 1
                        out['interactions'].append(
                            ('hbond', da.id, pa.id, hb, mid))

                # ── H-bonds: protein donor → drug acceptor ───────────────
                if pa.is_hbond_donor and da.is_hbond_acceptor:
                    hb = self.hydrogen_bond(pa.position, da.position)
                    out['hbonds'] += hb
                    if hb < -0.5:
                        out['n_hbonds'] += 1
                        out['interactions'].append(
                            ('hbond', da.id, pa.id, hb, mid))

                # ── Hydrophobic ──────────────────────────────────────────
                hp = self.hydrophobic(r, da, pa)
                out['hydrophobic'] += hp
                if hp < -0.3:
                    out['interactions'].append(
                        ('hydrophobic', da.id, pa.id, hp, mid))

        out['total'] = (out['vdw'] + out['electrostatic'] +
                        out['hbonds'] + out['hydrophobic'])
        return out


# ═══════════════════════════════════════════════════════════════════════════
# §6  MESH BUILDER  —  Agent 2  (3D Rendering)
# ═══════════════════════════════════════════════════════════════════════════

class MeshBuilder:
    """
    Builds PyVista meshes for atoms, covalent bonds, and interaction lines.
    Acts as the 'UI / Rendering Agent'.
    """

    @staticmethod
    def atom_sphere(atom: Atom, res: int = 18) -> pv.PolyData:
        """CPK-style sphere centred at atom.position with scaled VDW radius."""
        return pv.Sphere(
            radius=atom.radius,
            center=tuple(atom.position),
            theta_resolution=res,
            phi_resolution=res,
        )

    @staticmethod
    def bond_tube(pos_a: np.ndarray, pos_b: np.ndarray,
                  radius: float = BOND_RADIUS) -> pv.PolyData:
        """Solid cylinder connecting two bonded atoms."""
        pts  = np.array([pos_a, pos_b])
        line = pv.Spline(pts, 2)
        return line.tube(radius=radius)

    @staticmethod
    def dashed_tube(pos_a: np.ndarray, pos_b: np.ndarray,
                    n_dashes: int = 7,
                    radius: float = IACT_RADIUS) -> pv.PolyData:
        """
        Dashed-style tube for non-covalent interaction lines.
        Alternating short segments create the dashed appearance.
        """
        meshes = []
        for i in range(n_dashes):
            t0 = (2 * i)     / (2 * n_dashes)
            t1 = (2 * i + 1) / (2 * n_dashes)
            a  = pos_a + t0 * (pos_b - pos_a)
            b  = pos_a + t1 * (pos_b - pos_a)
            seg = pv.Spline(np.array([a, b]), 2).tube(radius=radius)
            meshes.append(seg)

        if not meshes:
            return pv.PolyData()
        combined = meshes[0]
        for m in meshes[1:]:
            combined = combined.merge(m)
        return combined


# ═══════════════════════════════════════════════════════════════════════════
# §7  INDUCED FIT ANIMATOR  —  Agent 3  (Conformational Change)
# ═══════════════════════════════════════════════════════════════════════════

class InducedFitAnimator:
    """
    Animates the protein's conformational change once binding energy
    crosses the threshold — the 'induced-fit' mechanism.

    Each protein atom smoothly shifts toward its nearest drug atom,
    simulating the binding pocket 'clamping' around the drug molecule.

    Easing function: smooth-step  f(t) = 3t² − 2t³
      → zero first-derivative at t=0 and t=1 → natural acceleration/deceleration

    Biology (Koshland, 1958 — Induced-Fit Model):
      The receptor is NOT a rigid lock. The drug's entry triggers
      physical reshaping of the binding site, optimising complementarity
      and creating a tighter, more specific interaction.
    """

    def __init__(self, protein_atoms: list, drug_atoms: list):
        self.protein_atoms = protein_atoms
        self.drug_atoms    = drug_atoms
        self.running       = False
        self._step         = 0
        self._total_steps  = 45   # ~1.8 s at 25 fps

    @staticmethod
    def smooth_step(t: float) -> float:
        """Smooth-step: f(0)=0, f(1)=1, f'(0)=f'(1)=0."""
        t = np.clip(t, 0.0, 1.0)
        return 3.0 * t**2 - 2.0 * t**3

    def _target(self, atom: Atom) -> np.ndarray:
        """
        Displacement target for one protein atom.
        Moves toward nearest drug atom by at most MAX_POCKET_SHIFT Å.
        """
        if not self.drug_atoms:
            return atom._origin.copy()

        dists = [np.linalg.norm(d.position - atom._origin)
                 for d in self.drug_atoms]
        idx  = int(np.argmin(dists))
        nearest_drug = self.drug_atoms[idx]
        d = dists[idx]

        if d < 0.1:
            return atom._origin.copy()

        direction = nearest_drug.position - atom._origin
        shift_mag = min(MAX_POCKET_SHIFT, 0.25 * d)
        unit_dir  = direction / (d + 1e-10)
        return atom._origin + shift_mag * unit_dir

    def step(self) -> bool:
        """
        Advance animation by one frame.
        Returns True while running, False when complete.
        """
        if self._step >= self._total_steps:
            self.running = False
            return False

        ease = self.smooth_step(self._step / self._total_steps)
        for atom in self.protein_atoms:
            tgt = self._target(atom)
            atom.position = atom._origin + ease * (tgt - atom._origin)

        self._step += 1
        return True

    def start(self):
        self._step   = 0
        self.running = True

    def reset(self):
        self.running = False
        self._step   = 0
        for atom in self.protein_atoms:
            atom.reset()


# ═══════════════════════════════════════════════════════════════════════════
# §8  MAIN APPLICATION  —  Orchestrator Agent
# ═══════════════════════════════════════════════════════════════════════════

class ProteinVisualizerApp:
    """
    Orchestrator Agent — owns the render window and coordinates all
    sub-agents in response to user input.

    Workflow on each slider change:
      1. Update drug atom positions              (position math)
      2. PhysicsEngine.score()                   (Agent 1)
      3. MeshBuilder renders updated drug        (Agent 2)
      4. MeshBuilder renders interaction lines   (Agent 2)
      5. Update HUD text                         (Agent 2)
      6. If threshold crossed → InducedFitAnim   (Agent 3)
    """

    def __init__(self):
        # ── Sub-agents ─────────────────────────────────────────────────
        self.physics  = PhysicsEngine()
        self.builder  = MeshBuilder()
        self.protein  = make_protein_atoms()
        self.drug     = make_drug_atoms()
        self.animator = InducedFitAnimator(self.protein, self.drug)

        # ── App state ──────────────────────────────────────────────────
        self._approach_t       = 0.0
        self._rotate_deg       = 0.0
        self._induced_fit_done = False
        self._atom_index       = {a.id: a for a in self.protein + self.drug}

        # ── Actor name registries ──────────────────────────────────────
        self._prot_actor_names = []
        self._drug_actor_names = []
        self._iact_actor_names = []
        self._hud_name         = 'hud_text'

        # ── Plotter ────────────────────────────────────────────────────
        pv.global_theme.background = '#0d1117'
        pv.global_theme.font.color = '#e6edf3'
        pv.global_theme.font.family = 'arial'

        self.p = pv.Plotter(
            title='Protein–Drug Binding Visualizer  |  Computational Biophysics',
            window_size=[1440, 900],
            lighting='three lights',
        )
        self.p.enable_anti_aliasing('ssaa')

    # ─── Rendering helpers ─────────────────────────────────────────────────

    def _add_atom(self, atom: Atom, registry: list,
                  metallic: float = 0.05, roughness: float = 0.55) -> str:
        name = f'atom_{atom.id}'
        self.p.add_mesh(
            self.builder.atom_sphere(atom),
            color=atom.color,
            smooth_shading=True,
            pbr=True, metallic=metallic, roughness=roughness,
            name=name,
        )
        registry.append(name)
        return name

    def _remove_registry(self, registry: list):
        for name in registry:
            self.p.remove_actor(name)
        registry.clear()

    def _add_protein_scene(self):
        """Render protein pocket (done once — it does not move)."""
        for atom in self.protein:
            self._add_atom(atom, self._prot_actor_names,
                           metallic=0.05, roughness=0.60)
            # Residue label above each atom
            label_pos = atom.position + np.array([0.0, atom.radius + 0.12, 0.0])
            self.p.add_point_labels(
                np.array([label_pos]), [atom.label],
                font_size=7, text_color='#6699AA',
                point_size=0, always_visible=True,
                name=f'lbl_{atom.id}',
            )

        for id1, id2 in PROTEIN_BONDS:
            a1 = self._atom_index.get(id1)
            a2 = self._atom_index.get(id2)
            if a1 and a2:
                self.p.add_mesh(
                    self.builder.bond_tube(a1.position, a2.position,
                                           BOND_RADIUS * 0.75),
                    color='#556677', name=f'pbond_{id1}_{id2}',
                )

    def _rebuild_drug_scene(self):
        """Remove and re-render drug atoms + covalent bonds."""
        self._remove_registry(self._drug_actor_names)

        for atom in self.drug:
            self._add_atom(atom, self._drug_actor_names,
                           metallic=0.20, roughness=0.40)

        for id1, id2 in DRUG_BONDS:
            a1 = self._atom_index.get(id1)
            a2 = self._atom_index.get(id2)
            if a1 and a2:
                name = f'dbond_{id1}_{id2}'
                self.p.add_mesh(
                    self.builder.bond_tube(a1.position, a2.position),
                    color='#AACCEE', name=name,
                )
                self._drug_actor_names.append(name)

    def _rebuild_protein_scene(self):
        """Re-render protein atoms at updated positions (induced-fit)."""
        for name in self._prot_actor_names:
            self.p.remove_actor(name)
        self._prot_actor_names.clear()
        for atom in self.protein:
            self._add_atom(atom, self._prot_actor_names,
                           metallic=0.05, roughness=0.60)

    def _rebuild_interactions(self, interactions: list):
        """Re-render all non-covalent interaction lines."""
        self._remove_registry(self._iact_actor_names)

        CLR = {
            'hbond':       CLR_HBOND,
            'ionic':       CLR_IONIC,
            'hydrophobic': CLR_HYDRO,
            'vdw':         CLR_VDW,
        }

        for i, (itype, d_id, p_id, energy, _mid) in enumerate(interactions):
            if itype == 'vdw' and abs(energy) < 0.5:
                continue  # skip weak vdW — avoids visual clutter

            da = self._atom_index.get(d_id)
            pa = self._atom_index.get(p_id)
            if not da or not pa:
                continue

            n_dash = 8 if itype == 'hbond' else 5
            tube   = self.builder.dashed_tube(
                da.position, pa.position,
                n_dashes=n_dash, radius=IACT_RADIUS,
            )
            name = f'iact_{i}'
            self.p.add_mesh(tube, color=CLR.get(itype, '#FFFFFF'),
                            opacity=0.85, name=name)
            self._iact_actor_names.append(name)

    def _update_hud(self, e: dict):
        """Refresh the energy heads-up display."""
        total = e['total']
        n_hb  = e['n_hbonds']

        if total > 0:
            status = '⚠  REPULSIVE — move drug closer'
            hcol   = 'tomato'
        elif total > -3:
            status = '◌  WEAK INTERACTION'
            hcol   = '#CCCC44'
        elif total > INDUCED_FIT_THRESHOLD:
            status = f'●  DOCKING...  ({total:.1f} / {INDUCED_FIT_THRESHOLD:.1f})'
            hcol   = 'cyan'
        else:
            status = '✦  BOUND  —  INDUCED FIT COMPLETE!'
            hcol   = 'lime'

        hud = (
            f"╔══ BINDING ENERGY ══════════════╗\n"
            f"║  ΔG total    :  {total:+7.2f} kcal/mol ║\n"
            f"╠════════════════════════════════╣\n"
            f"║  Van der Waals  {e['vdw']:+7.2f}         ║\n"
            f"║  Electrostatic  {e['electrostatic']:+7.2f}         ║\n"
            f"║  H-Bonds ({n_hb:1d})    {e['hbonds']:+7.2f}         ║\n"
            f"║  Hydrophobic    {e['hydrophobic']:+7.2f}         ║\n"
            f"╠════════════════════════════════╣\n"
            f"║  {status}\n"
            f"╚════════════════════════════════╝\n"
            f"\n"
            f" ── INTERACTION LEGEND ────────\n"
            f"  ╌╌  Cyan   = Hydrogen Bond\n"
            f"  ╌╌  Orange = Ionic / Electrostatic\n"
            f"  ╌╌  Green  = Hydrophobic\n"
            f"  ╌╌  Grey   = Van der Waals\n"
            f" ──────────────────────────────\n"
            f"  Induced-fit at: {INDUCED_FIT_THRESHOLD} kcal/mol"
        )

        self.p.remove_actor(self._hud_name)
        self.p.add_text(
            hud, position='upper_right',
            font_size=9, color=hcol, font='courier',
            name=self._hud_name,
        )

    # ─── Drug position math ────────────────────────────────────────────────

    def _place_drug(self, approach_t: float, rotate_deg: float):
        """
        Recompute drug atom positions from approach progress and rotation angle.

        approach_t  : [0.0, 1.0] — 0 = start (above pocket), 1 = docked
        rotate_deg  : [0, 360]   — rotation around the Y approach axis
        """
        rad     = np.radians(rotate_deg)
        cos_r   = np.cos(rad)
        sin_r   = np.sin(rad)
        Ry      = np.array([[cos_r, 0, sin_r],
                             [0,     1, 0    ],
                             [-sin_r,0, cos_r]])

        approach_vec = np.array([0.0, -APPROACH_DISTANCE, 0.0])
        centroid     = np.mean([a._origin for a in self.drug], axis=0)

        for atom in self.drug:
            rel             = atom._origin - centroid
            rotated         = Ry @ rel + centroid
            atom.position   = rotated + approach_t * approach_vec

    # ─── Main scene refresh ────────────────────────────────────────────────

    def _refresh(self):
        """
        Full scene refresh — called after every user interaction.

        Orchestration steps:
          1. Rebuild drug 3D mesh         (MeshBuilder)
          2. Calculate binding energy     (PhysicsEngine)
          3. Rebuild interaction lines    (MeshBuilder)
          4. Update HUD                   (MeshBuilder)
          5. Trigger induced fit if ready (InducedFitAnimator)
        """
        self._rebuild_drug_scene()

        energies = self.physics.score(self.drug, self.protein)
        self._rebuild_interactions(energies['interactions'])
        self._update_hud(energies)

        if (energies['total'] < INDUCED_FIT_THRESHOLD
                and not self._induced_fit_done):
            self._induced_fit_done = True
            self._trigger_induced_fit()

        self.p.render()

    # ─── Slider / button callbacks ─────────────────────────────────────────

    def _cb_approach(self, value: float):
        """Slider callback: move drug along approach vector."""
        self._approach_t = value / 10.0
        self._place_drug(self._approach_t, self._rotate_deg)
        self._refresh()

    def _cb_rotate(self, value: float):
        """Slider callback: rotate drug around approach axis."""
        self._rotate_deg = value
        self._place_drug(self._approach_t, self._rotate_deg)
        self._refresh()

    def _cb_reset(self, _flag):
        """Reset button: restore everything to the starting state."""
        self._approach_t       = 0.0
        self._rotate_deg       = 0.0
        self._induced_fit_done = False
        self.animator.reset()
        for atom in self.drug:
            atom.reset()
        self._place_drug(0.0, 0.0)
        self._rebuild_drug_scene()
        self._rebuild_protein_scene()
        self._rebuild_interactions([])
        self._update_hud({
            'vdw': 0.0, 'electrostatic': 0.0, 'hbonds': 0.0,
            'hydrophobic': 0.0, 'total': 0.0, 'n_hbonds': 0,
            'interactions': [],
        })
        self.p.render()

    # ─── Induced-fit animation ─────────────────────────────────────────────

    def _trigger_induced_fit(self):
        """
        Launches the induced-fit conformational change in a background thread.

        The protein atoms smoothly shift toward the docked drug (smooth-step
        easing), simulating the binding pocket 'closing around' the ligand.
        """
        self.animator.start()

        def _animate():
            while self.animator.step():
                self._rebuild_protein_scene()
                self.p.render()
                time.sleep(0.04)   # ≈ 25 fps

        threading.Thread(target=_animate, daemon=True).start()

    # ─── App launch ───────────────────────────────────────────────────────

    def run(self):
        """Construct the full scene, register UI widgets, and start the viewer."""

        # ── Fixed protein pocket ────────────────────────────────────────
        self._add_protein_scene()

        # ── Drug at starting position ───────────────────────────────────
        self._place_drug(0.0, 0.0)
        self._rebuild_drug_scene()

        # ── Pocket wireframe bounding box ───────────────────────────────
        self.p.add_mesh(
            pv.Box(bounds=(-4, 4, -3, 4, -3, 3)),
            style='wireframe', color='#1e3a4a',
            opacity=0.4, line_width=1, name='pocket_box',
        )

        # ── Coordinate axes ─────────────────────────────────────────────
        self.p.add_axes(
            line_width=2, color='#444444',
            xlabel='X (Å)', ylabel='Y (Å)', zlabel='Z (Å)',
            x_color='#FF4444', y_color='#44FF44', z_color='#4444FF',
            interactive=False,
        )

        # ── Title ───────────────────────────────────────────────────────
        self.p.add_text(
            'Protein–Drug Binding Visualizer\n'
            '  CPK: Grey=C  Blue=N  Red=O  Yellow=S',
            position='upper_left',
            font_size=9, color='#7EAABB', font='courier',
        )

        # ── Initial HUD ─────────────────────────────────────────────────
        self._update_hud({
            'vdw': 0.0, 'electrostatic': 0.0, 'hbonds': 0.0,
            'hydrophobic': 0.0, 'total': 0.0, 'n_hbonds': 0,
            'interactions': [],
        })

        # ── Slider: approach the pocket ─────────────────────────────────
        self.p.add_slider_widget(
            self._cb_approach,
            rng=[0.0, 10.0], value=0.0,
            title='▶  Approach Drug  (slide right to dock)',
            pointa=(0.03, 0.06), pointb=(0.50, 0.06),
            style='modern', color='cyan',
            pass_widget=False,
        )

        # ── Slider: rotate around Y axis ────────────────────────────────
        self.p.add_slider_widget(
            self._cb_rotate,
            rng=[0.0, 360.0], value=0.0,
            title='↻  Rotate Drug  (°)',
            pointa=(0.55, 0.06), pointb=(0.97, 0.06),
            style='modern', color='#99FF99',
            pass_widget=False,
        )

        # ── Reset button ────────────────────────────────────────────────
        self.p.add_checkbox_button_widget(
            self._cb_reset,
            value=False, position=(10, 95),
            size=30, border_size=2,
            color_on='#FF5555', color_off='#FF5555',
        )
        self.p.add_text('RESET', position=(48, 100),
                        font_size=8, color='#FF5555')

        # ── Camera ──────────────────────────────────────────────────────
        self.p.camera_position = [(0, -13, 9), (0, 1, 0), (0, 1, 0)]
        self.p.camera.zoom(1.15)

        # ── Console instructions ─────────────────────────────────────────
        print()
        print('═' * 62)
        print('  PROTEIN–DRUG BINDING VISUALIZER')
        print('  ──────────────────────────────────────────────────────')
        print('  ▶  Drag "Approach Drug" slider RIGHT to move the drug')
        print('     into the binding pocket.')
        print('  ↻  Drag "Rotate Drug" slider to change orientation.')
        print('  Watch non-covalent bonds appear in real time:')
        print('     Cyan   = Hydrogen bonds')
        print('     Orange = Ionic / electrostatic')
        print('     Green  = Hydrophobic contacts')
        print('     Grey   = Van der Waals')
        print(f'  ✦  Induced-fit animation triggers at '
              f'ΔG < {INDUCED_FIT_THRESHOLD} kcal/mol')
        print('  RESET button restores starting positions.')
        print('═' * 62)
        print()

        self.p.show()


# ═══════════════════════════════════════════════════════════════════════════
# §9  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app = ProteinVisualizerApp()
    app.run()
