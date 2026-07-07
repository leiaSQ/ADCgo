// Package mo loads the MO-coefficient / AO-overlap sidecar that accompanies an
// FCIDUMP. FCIDUMP carries neither, but the atom-resolved two-hole population
// (Tarantelli U-transform) needs the MO coefficients C and the AO overlap S, and
// the AO→atom map to define atomic groups. The sidecar is written by
// scripts/gen_fcidump.py.
package mo

import (
	"encoding/json"
	"fmt"
	"os"

	"adcgo/internal/adc/backend"
)

// Data is the parsed sidecar.
type Data struct {
	NAO       int
	NMO       int
	C         backend.Mat // nAO × nMO MO coefficients (row-major)
	S         backend.Mat // nAO × nAO AO overlap
	AOAtom    []int       // atom index per AO
	AtomNames []string    // atom labels, e.g. ["O","H1","H2"]
}

type sidecar struct {
	NAO       int         `json:"nao"`
	NMO       int         `json:"nmo"`
	MOCoeff   [][]float64 `json:"mo_coeff"`
	Overlap   [][]float64 `json:"overlap"`
	AOAtom    []int       `json:"ao_atom"`
	AtomNames []string    `json:"atom_names"`
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
	return &Data{NAO: s.NAO, NMO: s.NMO, C: c, S: sm, AOAtom: s.AOAtom, AtomNames: s.AtomNames}, nil
}
