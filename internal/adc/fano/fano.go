// Package fano computes electronic decay rates — Auger, ICD, ETMD and their
// second-order counterparts (double Auger, double ICD) — from an ADC secular matrix
// by the Fano/Feshbach method.
//
// Reference: P. Kolorenč and V. Averbukh, "Fano-ADC(2,2) method for electronic decay
// rates", J. Chem. Phys. 152, 214107 (2020), §III. The algorithm is the one
// implemented for the polarization propagator in ../ADC/adc2_pol
// (master_fano_new.f90, select_fano.f90, partgammas.f90, fspace.f90); that code is
// ADC(2)-extended over a 1h1p/2h2p space, not ADC(2,2), so what transfers is the
// algorithm, class for class: PP 1h1p -> IP 1h, PP 2h2p -> IP 2h1p, plus the new
// 3h2p class.
//
// # The partition
//
// The intermediate-state space is split into a bound subspace Q, which contains the
// discrete decaying state |Phi>, and a continuum subspace P, which contains the final
// states of the decay. The decay width follows from the coupling between them,
//
//	Gamma(E) = 2 pi |<Phi| H - E |chi_E>|^2,
//
// evaluated at E = E_Phi after Stieltjes imaging of the discrete pseudo-continuum.
//
// Note that ADCgo_plan.md's Track W text has P and Q swapped relative to the paper.
// This package follows the paper: Q is bound, P is the continuum.
//
// # Why there is no projected operator here
//
// Under the paper's scheme A (hole localization, §III B) the partition is a partition
// of the CONFIGURATION INDICES, so QMQ and PMP are literal sub-blocks of M. This
// package therefore does not wrap the ADC matrix in a projector: it hands the row
// subsets to Space.Restrict and gets back two ordinary ADC configuration spaces, over
// which sip.New assembles two ordinary ADC matrices. Every solver (Lanczos, Davidson,
// the low-memory driver), the matrix-free appliers and the GPU backends then work on
// them unchanged, and the cost is unchanged too — which is exactly the property the
// paper claims for scheme A. The reference does the same thing, building kpq_init and
// kpq_fin as two separate configuration spaces rather than projecting.
//
// The one quantity that is not inside either sub-block is the coupling between them.
// That needs no new machinery either: embedding |Phi> into the parent index space
// (zero outside Q) and applying the PARENT matrix gives (M Phi)_j = sum_{i in Q} M_ji
// Phi_i for every j, and reading off the P rows is the coupling vector exactly.
//
// Scheme B (adapted intermediate states, §III B 2) does need a basis change and will
// need a projected operator; it is a separate milestone and the interfaces here are
// shaped so it can be added without disturbing scheme A.
package fano

// Space is what this package needs of an ADC configuration space. *sip.Space
// satisfies it, and *dip.Space is intended to: the Fano layer is written against this
// interface rather than a concrete matrix so that extending it from single to double
// ionization is wiring rather than a rewrite.
//
// Rows are global indices in the operator's own ordering, 0 <= row < Size(), with the
// main (lowest) excitation class first.
type Space interface {
	// Size is the number of configurations, i.e. the matrix dimension.
	Size() int
	// MainBlockSize is the dimension of the main excitation class — 1h for single
	// ionization, 2h for double. Rows below it are main, rows above are satellites.
	MainBlockSize() int
	// Holes appends the occupied orbitals of row's configuration to dst and returns
	// the extended slice. Scheme A's Q/P criterion is a function of exactly this.
	Holes(row int, dst []int) []int
}
