// Package mo loads the MO-coefficient / AO-overlap sidecar that accompanies an
// FCIDUMP. FCIDUMP carries neither, but the atom-resolved two-hole population
// (Tarantelli U-transform) needs the MO coefficients C and the AO overlap S, and
// the AO→atom map to define atomic groups. It also carries no dipole integrals and
// no geometry, which the transition-moment machinery needs. The sidecar is written
// by scripts/fcidump/fcidump_common.py.
//
// The dipole and geometry keys are optional: sidecars written before they existed
// still load, with HasDipole false.
package mo

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/leiaSQ/ADCgo/backend"
)

// Data is the parsed sidecar.
type Data struct {
	NAO       int
	NMO       int
	C         backend.Mat // nAO × nMO MO coefficients (row-major)
	S         backend.Mat // nAO × nAO AO overlap
	AOAtom    []int       // atom index per AO
	AtomNames []string    // atom labels, e.g. ["O","H1","H2"]

	// HasDipole reports whether the dipole/geometry keys below were present.
	HasDipole   bool
	DipAO       [3]backend.Mat // nAO × nAO ⟨p|r_α|q⟩ about DipOrigin (a.u.)
	DipMO       [3]backend.Mat // nMO × nMO, = Cᵀ·DipAO·C, formed at load
	DipOrigin   [3]float64     // gauge origin (bohr): the centre of nuclear charge
	AtomCoords  [][3]float64   // bohr
	AtomCharges []float64      // nuclear charge Z_A
	// SCFDip is the whole-molecule RHF dipole (a.u., about DipOrigin) recorded by the
	// dumper. It is a gate value for the AO→MO transform, and GroundStateDipole
	// reproduces it only when C spans every occupied orbital — not for a frozen-core
	// active space.
	SCFDip [3]float64

	// Orbital labels (scripts/fcidump/orbitals.py). HasLabels reports whether the
	// group was present; it arrives all-or-none like the dipole keys.
	//
	// OrbKind[m] classifies MO m as occupied, compact virtual (assigned to an atom) or
	// free virtual (the diffuse/ghost complement). OrbAtom[m] is the owning atom for
	// occupied and compact MOs, as an index into AtomNames, and -1 for free ones.
	// GhostAtom[a] marks basis-only centres, which never own an orbital. Canonical is
	// false when the MOs were rotated away from the canonical HF orbitals, in which
	// case the FCIDUMP's Fock matrix is not diagonal and anything that rebuilds
	// orbital energies from its diagonal (mp.OrbitalEnergies) must not be used.
	HasLabels bool
	OrbKind   []OrbKind
	OrbAtom   []int
	GhostAtom []bool
	Canonical bool
}

// OrbKind classifies a molecular orbital of a labelled sidecar.
type OrbKind uint8

const (
	OrbOcc OrbKind = iota
	OrbCompact
	OrbFree
)

func (k OrbKind) String() string {
	switch k {
	case OrbOcc:
		return "occ"
	case OrbCompact:
		return "compact"
	default:
		return "free"
	}
}

type sidecar struct {
	NAO       int         `json:"nao"`
	NMO       int         `json:"nmo"`
	MOCoeff   [][]float64 `json:"mo_coeff"`
	Overlap   [][]float64 `json:"overlap"`
	AOAtom    []int       `json:"ao_atom"`
	AtomNames []string    `json:"atom_names"`

	DipAO       [][][]float64 `json:"dip_ao"`
	DipOrigin   []float64     `json:"dip_origin"`
	AtomCoords  [][]float64   `json:"atom_coords"`
	AtomCharges []float64     `json:"atom_charges"`
	SCFDip      []float64     `json:"scf_dip"`

	OrbKind   []string `json:"orb_kind"`
	OrbAtom   []int    `json:"orb_atom"`
	GhostAtom []bool   `json:"ghost_atoms"`
	Canonical *bool    `json:"canonical"`
}

// ReadFile parses the sidecar JSON at path.
//
// The decode streams off the file rather than going through os.ReadFile: the sidecar
// carries C, S and three nAO x nAO dipole matrices as JSON number text, so slurping
// it first held the raw bytes and the decoded [][]float64 at the same time — several
// times the size of the matrices themselves, at the front of every -mo run. Decoding
// from the file keeps only the decoder's buffer plus the result. Trailing content is
// still rejected, which json.Unmarshal did for free.
func ReadFile(path string) (*Data, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var s sidecar
	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	if err := dec.Decode(&s); err != nil {
		return nil, fmt.Errorf("mo: %w", err)
	}
	if err := dec.Decode(new(json.RawMessage)); err != io.EOF {
		return nil, fmt.Errorf("mo: trailing content after the sidecar object")
	}
	if len(s.MOCoeff) != s.NAO || len(s.Overlap) != s.NAO || len(s.AOAtom) != s.NAO {
		return nil, fmt.Errorf("mo: inconsistent dimensions (nao=%d)", s.NAO)
	}

	flat := func(rows [][]float64, r, c int) (backend.Mat, error) {
		m := backend.NewMat(r, c)
		for i, row := range rows {
			if len(row) != c {
				return m, fmt.Errorf("mo: row %d has %d cols, want %d", i, len(row), c)
			}
			copy(m.Data[i*c:(i+1)*c], row)
		}
		return m, nil
	}
	c, err := flat(s.MOCoeff, s.NAO, s.NMO)
	if err != nil {
		return nil, err
	}
	sm, err := flat(s.Overlap, s.NAO, s.NAO)
	if err != nil {
		return nil, err
	}
	d := &Data{NAO: s.NAO, NMO: s.NMO, C: c, S: sm, AOAtom: s.AOAtom, AtomNames: s.AtomNames,
		Canonical: true}
	if err := d.readDipole(&s, flat); err != nil {
		return nil, err
	}
	if err := d.readLabels(&s); err != nil {
		return nil, err
	}
	return d, nil
}

// readLabels decodes the optional orbital-label group (orb_kind, orb_atom,
// ghost_atoms, canonical). Like the dipole keys it is all or none, and every label is
// validated against the dimensions: a label that points at a ghost or past the atom
// list would otherwise surface much later as a wrong Q/P partition.
func (d *Data) readLabels(s *sidecar) error {
	present := 0
	for _, ok := range []bool{s.OrbKind != nil, s.OrbAtom != nil, s.GhostAtom != nil, s.Canonical != nil} {
		if ok {
			present++
		}
	}
	if present == 0 {
		return nil
	}
	if present != 4 {
		return fmt.Errorf("mo: sidecar has %d of the 4 orbital-label keys "+
			"(orb_kind, orb_atom, ghost_atoms, canonical); it must have all or none", present)
	}
	if len(s.OrbKind) != d.NMO || len(s.OrbAtom) != d.NMO {
		return fmt.Errorf("mo: orb_kind/orb_atom have %d/%d entries, want nmo=%d",
			len(s.OrbKind), len(s.OrbAtom), d.NMO)
	}
	if len(s.GhostAtom) != len(d.AtomNames) {
		return fmt.Errorf("mo: ghost_atoms has %d entries for %d atoms", len(s.GhostAtom), len(d.AtomNames))
	}
	d.OrbKind = make([]OrbKind, d.NMO)
	seenVirt := false
	for m, k := range s.OrbKind {
		switch k {
		case "occ":
			if seenVirt {
				return fmt.Errorf("mo: occupied MO %d after a virtual one", m)
			}
			d.OrbKind[m] = OrbOcc
		case "compact":
			d.OrbKind[m] = OrbCompact
			seenVirt = true
		case "free":
			d.OrbKind[m] = OrbFree
			seenVirt = true
		default:
			return fmt.Errorf("mo: orb_kind[%d] = %q, want occ, compact or free", m, k)
		}
		a := s.OrbAtom[m]
		if d.OrbKind[m] == OrbFree {
			if a != -1 {
				return fmt.Errorf("mo: free MO %d is labelled with atom %d", m, a)
			}
			continue
		}
		if a < 0 || a >= len(d.AtomNames) {
			return fmt.Errorf("mo: MO %d has atom %d outside 0..%d", m, a, len(d.AtomNames)-1)
		}
		if s.GhostAtom[a] {
			return fmt.Errorf("mo: MO %d is assigned to ghost centre %s", m, d.AtomNames[a])
		}
	}
	d.OrbAtom = s.OrbAtom
	d.GhostAtom = s.GhostAtom
	d.Canonical = *s.Canonical
	d.HasLabels = true
	return nil
}

// ReadCanonical is ReadFile for the ADC drivers: it refuses a sidecar that declares
// its orbitals non-canonical (a localized-orbital dump). Every ADC path reads
// orbital energies off the Fock diagonal and simplifies with a diagonal Fock matrix,
// which such a dump violates by O(0.1) Eh; those dumps are for the khci engine.
// Sidecars without the label group (every other dump) load exactly as before.
func ReadCanonical(path string) (*Data, error) {
	d, err := ReadFile(path)
	if err != nil {
		return nil, err
	}
	if d.HasLabels && !d.Canonical {
		return nil, fmt.Errorf("mo: %s declares canonical=false (a localized-orbital dump); "+
			"the ADC drivers need canonical HF orbitals", path)
	}
	return d, nil
}

// NOccLabelled is the number of MOs labelled occupied (they come first).
func (d *Data) NOccLabelled() int {
	n := 0
	for n < len(d.OrbKind) && d.OrbKind[n] == OrbOcc {
		n++
	}
	return n
}

// readDipole decodes the optional dipole/geometry keys. They arrive as a set: a
// sidecar that has some but not all of them was written by something that got the
// contract wrong, and silently loading a half-populated Data would surface later as a
// wrong transition moment rather than as a parse error.
func (d *Data) readDipole(s *sidecar, flat func([][]float64, int, int) (backend.Mat, error)) error {
	present := 0
	for _, ok := range []bool{s.DipAO != nil, s.DipOrigin != nil, s.AtomCoords != nil, s.AtomCharges != nil} {
		if ok {
			present++
		}
	}
	if present == 0 {
		return nil // a pre-dipole sidecar; the legacy consumers need nothing more
	}
	if present != 4 {
		return fmt.Errorf("mo: sidecar has %d of the 4 dipole/geometry keys "+
			"(dip_ao, dip_origin, atom_coords, atom_charges); it must have all or none", present)
	}
	if len(s.DipAO) != 3 || len(s.DipOrigin) != 3 {
		return fmt.Errorf("mo: dip_ao has %d components and dip_origin %d, want 3 and 3",
			len(s.DipAO), len(s.DipOrigin))
	}
	if len(s.AtomCoords) != len(s.AtomCharges) {
		return fmt.Errorf("mo: %d atom_coords but %d atom_charges",
			len(s.AtomCoords), len(s.AtomCharges))
	}
	for a := range 3 {
		if len(s.DipAO[a]) != d.NAO {
			return fmt.Errorf("mo: dip_ao[%d] has %d rows, want nao=%d", a, len(s.DipAO[a]), d.NAO)
		}
		m, err := flat(s.DipAO[a], d.NAO, d.NAO)
		if err != nil {
			return err
		}
		d.DipAO[a] = m
		d.DipMO[a] = TransformMO(m, d.C)
		d.DipOrigin[a] = s.DipOrigin[a]
	}
	d.AtomCoords = make([][3]float64, len(s.AtomCoords))
	for a, r := range s.AtomCoords {
		if len(r) != 3 {
			return fmt.Errorf("mo: atom_coords[%d] has %d components, want 3", a, len(r))
		}
		d.AtomCoords[a] = [3]float64{r[0], r[1], r[2]}
	}
	d.AtomCharges = s.AtomCharges
	if len(s.SCFDip) == 3 {
		d.SCFDip = [3]float64{s.SCFDip[0], s.SCFDip[1], s.SCFDip[2]}
	}
	d.HasDipole = true
	return nil
}
