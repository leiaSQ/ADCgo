// Package mo loads the MO-coefficient / AO-overlap sidecar that accompanies an
// FCIDUMP. FCIDUMP carries neither, but the atom-resolved two-hole population
// (Tarantelli U-transform) needs the MO coefficients C and the AO overlap S, and
// the AO→atom map to define atomic groups. It also carries no dipole integrals and
// no geometry, which the transition-moment machinery needs. The sidecar is written
// by scripts/fcidump_common.py.
//
// The dipole and geometry keys are optional: sidecars written before they existed
// still load, with HasDipole false.
package mo

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
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
}

// ReadFile parses the sidecar JSON at path.
func ReadFile(path string) (*Data, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s sidecar
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, fmt.Errorf("mo: %w", err)
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
	d := &Data{NAO: s.NAO, NMO: s.NMO, C: c, S: sm, AOAtom: s.AOAtom, AtomNames: s.AtomNames}
	if err := d.readDipole(&s, flat); err != nil {
		return nil, err
	}
	return d, nil
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
