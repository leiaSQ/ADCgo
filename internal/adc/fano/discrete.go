package fano

import (
	"fmt"
	"slices"

	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
)

// discrete.go — selecting the decaying state |Phi> from the QMQ spectrum.

// Discrete is the bound state of the Feshbach partition: the QMQ eigenstate that
// represents the initially created vacancy, and whose width this whole package exists
// to compute.
type Discrete struct {
	// Energy is E_Phi in hartree — the position of the decaying state, and the energy
	// at which the Stieltjes-imaged Gamma(E) is finally evaluated.
	Energy float64
	// Vec is Phi in Q-subspace coordinates (length Partition.QSize()).
	Vec []float64
	// Weight is Phi's squared weight on the main-class configurations carrying the
	// vacancy — the criterion it was selected by.
	Weight float64
	// Root is which QMQ root it is, 0-based in ascending energy.
	Root int
	// Rows are the Q-subspace main-class rows the weight was summed over.
	Rows []int
}

// SelectDiscrete picks |Phi> out of a solved QMQ spectrum.
//
// Among the roots whose squared weight on the vacancy's main-class configurations
// reaches minWeight, it returns the nth in ascending energy (nth is 0-based). qsp is
// the RESTRICTED Q space, and res must have been solved over it with
// lanczos.Options.WantFull, since Phi's satellite components are what the coupling is
// built from.
//
// This mirrors get_bound (../ADC/adc2_pol/fspace.f90:400) together with
// get_ncnfi_ryd and select_fstate_ryd (select_fano.f90:1280, :1310): the reference
// sums arr(ncnfi_ryd(i),:)**2 over the main-class configurations whose hole is hinit,
// keeps the roots at or above mspacewi, sorts those by energy and indexes with
// ninista. minWeight here is mspacewi and nth is ninista-1.
//
// For single ionization the main class is 1h, so the sum is over the single row whose
// hole is the vacancy — the paper's "selected by its leading 1h configuration". The
// sum over rows is not redundant generality: for double ionization the main class is
// 2h and every 2h configuration carrying the vacancy contributes, which is also the
// shape the reference has (its main class is 1h1p, one row per particle orbital).
func SelectDiscrete(qsp Space, res lanczos.Result, vacancy, nth int, minWeight float64) (Discrete, error) {
	if !res.HasFull() {
		return Discrete{}, fmt.Errorf("fano: the QMQ solve did not retain full Ritz vectors; " +
			"|Phi> needs its satellite components, so solve Q with lanczos.Options.WantFull")
	}
	if got, want := res.FullVecs.Rows, qsp.Size(); got != want {
		return Discrete{}, fmt.Errorf("fano: the QMQ Ritz vectors have %d rows but the Q space "+
			"has %d configurations; the solve and the partition disagree", got, want)
	}
	if nth < 0 {
		return Discrete{}, fmt.Errorf("fano: negative root index %d", nth)
	}

	// The Q-space main-class rows carrying the vacancy.
	var rows []int
	holes := make([]int, 0, 3)
	for r := range qsp.MainBlockSize() {
		if slices.Contains(qsp.Holes(r, holes[:0]), vacancy) {
			rows = append(rows, r)
		}
	}
	if len(rows) == 0 {
		return Discrete{}, fmt.Errorf("fano: no main-class configuration of the bound subspace "+
			"carries a hole in orbital %d (Q has %d main-class rows); either the initial vacancy "+
			"is not in Q, or it is not an occupied orbital of this sector's irrep",
			vacancy, qsp.MainBlockSize())
	}

	// res.Values is ascending, so scanning it in order is already the reference's
	// energy sort of the qualifying roots.
	found := 0
	best, bestW := -1, 0.0
	for j := range res.Values {
		var w float64
		for _, r := range rows {
			c := res.FullVecs.At(r, j)
			w += c * c
		}
		if w > bestW {
			best, bestW = j, w
		}
		if w < minWeight {
			continue
		}
		if found == nth {
			vec := make([]float64, qsp.Size())
			for i := range vec {
				vec[i] = res.FullVecs.At(i, j)
			}
			return Discrete{
				Energy: res.Values[j], Vec: vec, Weight: w, Root: j, Rows: rows,
			}, nil
		}
		found++
	}
	if found == 0 {
		return Discrete{}, fmt.Errorf("fano: no QMQ root of %d has weight >= %g on the vacancy "+
			"configuration(s) %v; the strongest is root %d at %.4g. Lower -fano-wmin, or solve "+
			"more roots — the discrete state may lie above the ones converged",
			len(res.Values), minWeight, rows, best, bestW)
	}
	return Discrete{}, fmt.Errorf("fano: only %d QMQ root(s) reach weight %g on the vacancy "+
		"configuration(s) %v, so there is no root number %d", found, minWeight, rows, nth)
}
