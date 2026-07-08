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
    &active
      frozen-core 1              # optional
      active 2 to 30             # optional; omit for the full MO space
    &output
      fcidump  h2o_dzp.fcidump
      sidecar  h2o_dzp.mo.json   # optional
      manifest h2o_dzp.ref.json  # optional
    &adc                         # optional; consumed by adcgo_run.sh only
      args   -dip -order 2 -solver lanczos -spin both
      sym    1 to 4              # loop target irreps, or `all`

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
    basis_name: str = None
    cartesian: bool = False
    # scf
    charge: int = 0
    spin: int = 0                   # 2S
    symmetry: object = True         # True (auto) | False (off) | group name str
    gate: float = None
    conv_tol: float = 1e-12
    conv_tol_grad: float = 1e-9
    # orbital selection
    frozen_core: int = None
    active: str = None
    # output
    fcidump: str = None
    sidecar: str = None
    manifest: str = None
    # adc (driver only)
    adc_args: str = ""
    adc_sym: str = "all"
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
        _apply(cfg, section, key, val)

    _validate(cfg)
    return cfg


def _apply(cfg, section, key, val):
    if section == "geometry":
        if key == "file":
            cfg.geom_file = val
        elif key == "unit":
            cfg.unit = val
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
        else:
            raise ValueError(f"unknown &scf key {key!r}")
    elif section == "active":
        if key in ("frozen-core", "frozen_core", "core"):
            cfg.frozen_core = int(val)
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
