// Package mp implements the Møller–Plesset ground state that the ADC
// intermediate-state representation is built on. For the M0 spike this is the
// closed-shell RHF-MP2 correlation energy, plus reconstruction of the canonical
// orbital energies from the Fock diagonal (FCIDUMP does not store them).
package mp

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

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

// MaxFockOffDiagonal is the largest |f_pq|, p != q, of the Fock matrix rebuilt from
// the FCIDUMP, f_pq = h_pq + Σ_{i∈occ} [ 2 (pq|ii) − (pi|iq) ]. It is ~ the SCF
// gradient (1e-9) for canonical HF orbitals and O(0.1) Eh for a rotated basis such as
// localized orbitals (scripts/fcidump/orbitals.py), where the diagonal
// OrbitalEnergies returns is NOT a set of orbital energies. Parallel over rows.
func MaxFockOffDiagonal(d *fcidump.Data, nocc int) float64 {
	n := d.NORB
	W := parallel.ChunkWorkers(n)
	worst := make([]float64, W)
	parallel.Chunks(n, W, func(w, lo, hi int) {
		for p := lo; p < hi; p++ {
			for q := 0; q < n; q++ {
				if q == p {
					continue
				}
				f := d.OneE(p, q)
				for i := 0; i < nocc; i++ {
					f += 2*d.TwoE(p, q, i, i) - d.TwoE(p, i, i, q)
				}
				if f < 0 {
					f = -f
				}
				if f > worst[w] {
					worst[w] = f
				}
			}
		}
	})
	var m float64
	for _, x := range worst {
		m = max(m, x)
	}
	return m
}

// CanonicalTol is the largest off-diagonal Fock element RequireCanonical accepts:
// four orders above a converged SCF (conv_tol_grad 1e-9), four below a rotated basis.
const CanonicalTol = 1e-5

// RequireCanonical errors when the FCIDUMP's orbitals are not canonical HF orbitals.
//
// It is a diagnostic, deliberately not called by OrbitalEnergies or the drivers: the
// theADCcode matched-integral tape (testdata/reference/h2o_dzp.matched.fcidump) has
// off-diagonal Fock elements up to 1.03 Eh by construction and is used
// semi-canonically, diagonal only, by bit-exact gates. The drivers instead refuse a
// dump whose MO sidecar declares canonical=false (mo.ReadCanonical).
func RequireCanonical(d *fcidump.Data, nocc int) error {
	if m := MaxFockOffDiagonal(d, nocc); m > CanonicalTol {
		return fmt.Errorf("mp: the FCIDUMP orbitals are not canonical (max |f_pq|, p != q, "+
			"= %.3e Eh > %.0e): orbital energies cannot be read off the Fock diagonal. A "+
			"localized-orbital dump (&orbitals localized) is for the khci engine, not for ADC", m, CanonicalTol)
	}
	return nil
}

// mp2Chunks is the number of static (i,j) ranges MP2Corr reduces over.
//
// It is a CONSTANT, not GOMAXPROCS, on purpose: it fixes the summation grouping, so
// the energy a run reports depends only on the molecule and not on how many cores
// the node happened to have. The work per (i,j) pair is exactly nvir^2, so a static
// split needs no extra chunks for balance; 512 just keeps every core fed at
// the production system's nocc^2 = 3364 pairs.
const mp2Chunks = 512

// MP2Corr returns the closed-shell RHF-MP2 correlation energy:
//
//	E2 = Σ_{ij∈occ} Σ_{ab∈vir} (ia|jb)[2(ia|jb) − (ib|ja)] / (ε_i+ε_j−ε_a−ε_b)
//
// The (i,j) occupied pairs are split into mp2Chunks static ranges, each summed on
// its own goroutine, and the partials are added back in ascending chunk order. At
// production scale this is ~8e7 iterations (nocc^2*nvir^2 = 3364*23716), each two
// random lookups into the 16.2 GB ERI plus a divide — latency-bound work that was
// running on one core.
//
// THIS CHANGES THE SUMMATION ORDER. The inner a,b loops and the order of pairs
// within a chunk are untouched, but the running total is now reassociated at the 512
// chunk boundaries, so E2 can differ from the old value in the last few ulp. It is
// NOT a silent reassociation: the split is static (parallel.Chunks, not a
// work-stealing pool), the chunk count is a constant rather than the core count, and
// the partials are combined in fixed ascending order — so the result is reproducible
// bit-for-bit run to run and node to node. Measured drift on the h2o test dump is
// 3.9e-16 Ha on a correlation energy of -0.204 Ha (1.9e-15 relative); the production system has ~4
// orders more terms, so expect ~1e-14 Ha. That is six orders below the 1e-8 Ha
// agreement with pyscf that TestMP2Corr gates, and TestMP2CorrChunkedReduction pins
// it against the serial-order sum.
func MP2Corr(d *fcidump.Data, nocc int, eps []float64) float64 {
	n := d.NORB
	npair := nocc * nocc
	chunks := min(npair, mp2Chunks)
	if chunks < 1 {
		chunks = 1
	}
	partial := make([]float64, chunks)
	parallel.Chunks(npair, chunks, func(w, lo, hi int) {
		var e2 float64
		for ij := lo; ij < hi; ij++ {
			i, j := ij/nocc, ij%nocc
			for a := nocc; a < n; a++ {
				for b := nocc; b < n; b++ {
					iajb := d.TwoE(i, a, j, b)
					ibja := d.TwoE(i, b, j, a)
					denom := eps[i] + eps[j] - eps[a] - eps[b]
					e2 += iajb * (2*iajb - ibja) / denom
				}
			}
		}
		partial[w] = e2
	})
	var e2 float64
	for w := range chunks { // fixed ascending order: deterministic reduction
		e2 += partial[w]
	}
	return e2
}
