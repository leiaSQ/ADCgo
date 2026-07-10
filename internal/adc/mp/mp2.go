// Package mp implements the Møller–Plesset ground state that the ADC
// intermediate-state representation is built on. For the M0 spike this is the
// closed-shell RHF-MP2 correlation energy, plus reconstruction of the canonical
// orbital energies from the Fock diagonal (FCIDUMP does not store them).
package mp

import "github.com/leiaSQ/ADCgo/internal/adc/fcidump"

// NOcc returns the number of doubly occupied spatial orbitals for a
// closed-shell (RHF) reference.
func NOcc(d *fcidump.Data) int { return d.NELEC / 2 }

// OrbitalEnergies reconstructs the canonical HF orbital energies from the Fock
// diagonal in the MO basis:
//
//	f_pp = h_pp + Σ_{i∈occ} [ 2 (pp|ii) − (pi|ip) ]
//
// In a canonical HF MO basis the Fock matrix is diagonal, so f_pp = ε_p.
func OrbitalEnergies(d *fcidump.Data, nocc int) []float64 {
	n := d.NORB
	eps := make([]float64, n)
	for p := 0; p < n; p++ {
		e := d.OneE(p, p)
		for i := 0; i < nocc; i++ {
			e += 2*d.TwoE(p, p, i, i) - d.TwoE(p, i, i, p)
		}
		eps[p] = e
	}
	return eps
}

// MP2Corr returns the closed-shell RHF-MP2 correlation energy:
//
//	E2 = Σ_{ij∈occ} Σ_{ab∈vir} (ia|jb)[2(ia|jb) − (ib|ja)] / (ε_i+ε_j−ε_a−ε_b)
func MP2Corr(d *fcidump.Data, nocc int, eps []float64) float64 {
	n := d.NORB
	var e2 float64
	for i := 0; i < nocc; i++ {
		for j := 0; j < nocc; j++ {
			for a := nocc; a < n; a++ {
				for b := nocc; b < n; b++ {
					iajb := d.TwoE(i, a, j, b)
					ibja := d.TwoE(i, b, j, a)
					denom := eps[i] + eps[j] - eps[a] - eps[b]
					e2 += iajb * (2*iajb - ibja) / denom
				}
			}
		}
	}
	return e2
}
