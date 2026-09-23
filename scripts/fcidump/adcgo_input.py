"""Sectioned ADCgo dump input file -> Config.

Mirrors theADCcode's ``&``-delimited stdin file (../ADC/regression_h2o/scf_adc.in)
but for the *dump* step: it ties geometry + basis + SCF + orbital selection + output
into one file that ``dump_fcidump.py`` consumes, with an optional ``&adc`` section the
end-to-end driver (``adcgo_run.sh``) uses to invoke the ``adcgo`` binary.

Grammar (``#`` starts a comment; blank lines ignored):

    &geometry
      file   _zmatrix.dat        # GAMESS-UK zmat, .xyz, or Cartesian list
      unit   angstrom            # optional override (zmat carries its own)
    &basis
      file   _basisset.dat       # OR:  name cc-pvdz
      cartesian on               # GAMESS-UK uses cartesian GTOs
    &scf
      charge 0
      spin   0                   # 2S (number of unpaired electrons)
      symmetry auto              # auto | off | C2v | Cs | ...
      gate   -76.0498071428      # optional SCF-energy gate
      max_cycle 200              # optional; pyscf's 50 can be short of conv_tol=1e-12
    &active
      frozen-core 1              # optional: freeze the N lowest MOs
      frozen-list 2 to 6         # optional: freeze an explicit (non-lowest) MO set
      active 2 to 30             # optional; omit for the full MO space
    &output
      fcidump  h2o_dzp.fcidump
      sidecar  h2o_dzp.mo.json   # optional
      manifest h2o_dzp.ref.json  # optional
    &adc                         # optional; consumed by adcgo_run.sh only
      args   -dip -order 2 -solver lanczos -spin both
      sym    1 to 4              # loop target irreps, or `all`
    &ghost_sites                 # optional: basis-only centres (continuum / Rydberg sets)
      unit   angstrom            # optional, default angstrom
      GT     0.0 0.0 0.0  kbj:2:1-8           # label x y z basis-spec [basis-spec...]
      GCOM   0.0 1.5 0.0  kbj:1:6-10 aug-cc-pvdz@He
    &orbitals                    # optional: rotated MO basis (orbitals.py)
      scheme        localized    # canonical (default) | localized
      compact_basis cc-pvdz      # parent basis whose shells count as compact
      compact_thresh 0.02        # projected-overlap eigenvalue kept as compact
      one_per_atom  on           # every real atom owns exactly one occupied orbital
      lindep        1e-6         # canonical orthogonalization threshold

Ghost-site basis specs are "kbj:<lmax>:<n1>-<n2>" (Kaufmann-Baumeister-Jungen
continuum exponents, kbj.py) or "<basis>@<element>". The localized scheme
writes a NON-canonical FCIDUMP (localized occupied, per-atom compact virtuals, free
complement) and labels every MO in the sidecar (orb_kind, orb_atom, ghost_atoms,
canonical=false); it cannot be combined with &active.

Keys are the first token on a line; the value is the rest of the line (so multi-word
values like `active 2 to 30` or `args -dip -order 2` work). Relative file paths
resolve against the input file's directory.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    base_dir: str = "."
    # geometry
    geom_file: str = None
    unit: str = None                # optional override
    # basis
    basis_file: str = None
    # ghost: 1-based atom indices (GAMESS "A to B" syntax) kept as BASIS CENTRES ONLY -- no
    # nucleus, no electrons. This is the counterpoise construction: ghosting one fragment of a
    # complex leaves the other in the FULL complex basis, so its energy carries the same basis
    # set superposition error the complex does and the two are comparable.
    ghost: str = None
    basis_name: str = None
    cartesian: bool = False
    # scf
    charge: int = 0
    spin: int = 0                   # 2S
    symmetry: object = True         # True (auto) | False (off) | group name str
    gate: float = None
    conv_tol: float = 1e-12
    conv_tol_grad: float = 1e-9
    max_cycle: int = 50             # pyscf's default; raise it for slow-converging SCF
    # Convergence aids. They exist for NON-COVALENT systems: on a pi-stacked dimer the two
    # fragments contribute near-degenerate frontier orbitals, so plain DIIS from a minao guess
    # oscillates between them, and can settle on a charge-transfer solution that looks
    # converged and is physically wrong. Defaults are pyscf's, so every existing deck is
    # unchanged.
    init_guess: str = None          # minao (pyscf default) | atom | huckel | 1e | chkfile
    level_shift: float = 0.0        # Ha added to the virtual diagonal; decays as it converges
    damp: float = 0.0               # Fock damping factor for the early cycles
    soscf: bool = False             # second-order (Newton) SCF after DIIS stalls
    # orbital selection
    frozen_core: int = None
    frozen_list: str = None
    active: str = None
    # output
    fcidump: str = None
    sidecar: str = None
    manifest: str = None
    # adc (driver only)
    adc_args: str = ""
    adc_sym: str = "all"
    # ghost sites: list of {"label", "xyz" (Angstrom), "basis": [specs]}
    ghost_sites: list = field(default_factory=list)
    ghost_unit: str = "angstrom"
    # orbital scheme
    orbitals: str = "canonical"
    compact_basis: str = None
    compact_thresh: float = 0.02
    one_per_atom: bool = True
    lindep: float = None
    # provenance
    meta: dict = field(default_factory=dict)

    def resolve(self, path):
        """Resolve a possibly-relative path against the input file's directory."""
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(self.base_dir, path)


_TRUE = {"on", "true", "yes", "1"}
_FALSE = {"off", "false", "no", "0", "none"}


def _to_bool(val):
    low = val.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(f"expected a boolean (on/off), got {val!r}")


def _to_symmetry(val):
    low = val.strip().lower()
    if low in ("auto", "on", "true", "detect"):
        return True
    if low in ("off", "false", "none", "no"):
        return False
    return val.strip()          # explicit point-group name, e.g. C2v


def parse_input_file(path):
    """Parse a sectioned input file into a :class:`Config`."""
    with open(path) as fh:
        text = fh.read()
    cfg = Config(base_dir=os.path.dirname(os.path.abspath(path)))

    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("&"):
            section = line.strip()[1:].lower()
            continue
        if section is None:
            raise ValueError(f"key/value before any &section: {line!r}")
        stripped = line.strip()
        parts = stripped.split(None, 1)
        key = parts[0].lower()
        val = parts[1].strip() if len(parts) > 1 else ""
        if section == "ghost_sites" and key != "unit":
            key = parts[0]  # a site label is a name (charge rules refer to it): keep case
        _apply(cfg, section, key, val)

    _validate(cfg)
    return cfg


def _apply(cfg, section, key, val):
    if section == "geometry":
        if key == "file":
            cfg.geom_file = val
        elif key == "unit":
            cfg.unit = val
        elif key == "ghost":
            cfg.ghost = val
        else:
            raise ValueError(f"unknown &geometry key {key!r}")
    elif section == "basis":
        if key == "file":
            cfg.basis_file = val
        elif key == "name":
            cfg.basis_name = val
        elif key == "cartesian":
            cfg.cartesian = _to_bool(val)
        else:
            raise ValueError(f"unknown &basis key {key!r}")
    elif section == "scf":
        if key == "charge":
            cfg.charge = int(val)
        elif key == "spin":
            cfg.spin = int(val)
        elif key == "symmetry":
            cfg.symmetry = _to_symmetry(val)
        elif key == "gate":
            cfg.gate = float(val)
        elif key in ("conv_tol", "conv"):
            cfg.conv_tol = float(val)
        elif key == "conv_tol_grad":
            cfg.conv_tol_grad = float(val)
        elif key in ("max_cycle", "max-cycle", "maxcycle"):
            cfg.max_cycle = int(val)
        elif key in ("init_guess", "init-guess", "guess"):
            allowed = ("minao", "atom", "huckel", "1e", "chkfile")
            if val not in allowed:
                raise ValueError(f"&scf init_guess {val!r} not one of {allowed}")
            cfg.init_guess = val
        elif key in ("level_shift", "level-shift", "levelshift"):
            cfg.level_shift = float(val)
        elif key == "damp":
            cfg.damp = float(val)
        elif key == "soscf":
            cfg.soscf = _to_bool(val)
        else:
            raise ValueError(f"unknown &scf key {key!r}")
    elif section == "active":
        if key in ("frozen-core", "frozen_core", "core"):
            cfg.frozen_core = int(val)
        elif key in ("frozen-list", "frozen_list", "core-list"):
            cfg.frozen_list = val
        elif key == "active":
            cfg.active = val
        else:
            raise ValueError(f"unknown &active key {key!r}")
    elif section == "output":
        if key == "fcidump":
            cfg.fcidump = val
        elif key == "sidecar":
            cfg.sidecar = val
        elif key == "manifest":
            cfg.manifest = val
        else:
            raise ValueError(f"unknown &output key {key!r}")
    elif section == "ghost_sites":
        if key == "unit":
            if val.lower() not in ("angstrom", "bohr"):
                raise ValueError(f"&ghost_sites unit {val!r}: want angstrom or bohr")
            cfg.ghost_unit = val.lower()
            return
        fields = val.split()
        if len(fields) < 4:
            raise ValueError(f"&ghost_sites line for {key!r} wants x y z basis-spec...")
        try:
            xyz = [float(x) for x in fields[:3]]
        except ValueError:
            raise ValueError(f"&ghost_sites {key!r}: bad coordinates {fields[:3]}")
        if any(g["label"] == key for g in cfg.ghost_sites):
            raise ValueError(f"&ghost_sites label {key!r} given twice")
        cfg.ghost_sites.append({"label": key, "xyz": xyz, "basis": fields[3:]})
    elif section == "orbitals":
        if key == "scheme":
            if val not in ("canonical", "localized"):
                raise ValueError(f"&orbitals scheme {val!r}: want canonical or localized")
            cfg.orbitals = val
        elif key == "compact_basis":
            cfg.compact_basis = val
        elif key == "compact_thresh":
            cfg.compact_thresh = float(val)
        elif key == "one_per_atom":
            cfg.one_per_atom = _to_bool(val)
        elif key == "lindep":
            cfg.lindep = float(val)
        else:
            raise ValueError(f"unknown &orbitals key {key!r}")
    elif section == "adc":
        if key == "args":
            cfg.adc_args = val
        elif key == "sym":
            cfg.adc_sym = val
        else:
            raise ValueError(f"unknown &adc key {key!r}")
    else:
        raise ValueError(f"unknown section &{section}")


def _validate(cfg):
    if not cfg.geom_file:
        raise ValueError("&geometry file is required")
    if not cfg.basis_file and not cfg.basis_name:
        raise ValueError("&basis needs either `file` or `name`")
    if cfg.basis_file and cfg.basis_name:
        raise ValueError("&basis: give `file` or `name`, not both")
    if not cfg.fcidump:
        raise ValueError("&output fcidump is required")
    if cfg.orbitals == "localized" and (cfg.active or cfg.frozen_core or cfg.frozen_list):
        raise ValueError("&orbitals localized cannot be combined with &active: the "
                         "localized basis rotates the full occupied and virtual spaces")
