package sip

import "math"

// spincsf.go — genealogical (Kotani branching-diagram) spin functions for the
// open shells of an ADC configuration.
//
// The ISR-ADC(2,2) working equations of Kolorenč/Averbukh, J. Chem. Phys. 152,
// 214107 (2020), Appendix A2-A22, are given for *spin-orbital* configurations
// ("primitive" excitations, their Eq. 9). ADCgo's secular matrix is spin-adapted
// to the doublet of the cation, so every appendix element has to be contracted
// with the coefficients that expand a spin-adapted configuration over primitive
// ones. The appendix says as much:
//
//	"For an efficient implementation, it is necessary to generate spin-free
//	 working equations for the total spin value of interest (typically S = 1/2 in
//	 order to couple with the 1h states). This is achieved by standard angular
//	 momentum algebra techniques."
//
// Rather than transcribe a coefficient table per block (the route taken for CVS
// ADC(4) in coeff4.go, where the tables come verbatim from the reference's
// init{0,1,2}.F), this file derives the coefficients. That matters most for the
// 3h2p class, where a configuration carries up to five spin functions — deriving
// those by hand for every block of A15-A22 would be far more error-prone than
// generating them from the branching diagram and gating the generator.
//
// The construction is the standard genealogical one: couple the open shells in a
// fixed order, one spin-1/2 at a time, so that an intermediate total spin S_t is
// assigned after each shell. A "path" S_0=0, S_1, ..., S_n with S_n = 1/2 labels
// one doublet spin function, and its expansion coefficient over a determinant is
// the product of the Clebsch-Gordan coefficients along the path. Paths are the
// spin-function index: this is exactly the label Config.Typ (2 functions for a
// 2h1p with k != l) and Config3.Spin (1, 2 or 5 functions, cf. maxS3) already
// carry.
//
// Everything here is in terms of "open shells" — singly occupied spatial
// orbitals of the (N-1)-electron determinant — so it is independent of which ADC
// class the configuration belongs to. The mapping from an ADC configuration to
// its open shells, and the operator-ordering sign that goes with it, lives in
// spinadapt.go.

// spinUp/spinDown are the two values of a determinant's spin string entry. A
// string is indexed by open shell, in the order the caller fixed.
const (
	spinUp   = 0
	spinDown = 1
)

// csfBasis is the doublet spin-function basis for n open shells: the M_s = +1/2
// determinants (dets) and one coefficient row per spin function (paths).
//
// Coef[p][d] is the coefficient of determinant Dets[d] in spin function p. The
// rows are orthonormal, and their number is the number of S = 1/2 branching
// paths, which is C(n, (n-1)/2) - C(n, (n-3)/2) — 1 for n = 1, 2 for n = 3, 5
// for n = 5, matching maxS3.
type csfBasis struct {
	N     int     // open shells
	Dets  [][]int // M_s = +1/2 spin strings, each of length N, in ascending order
	Coef  [][]float64
	paths [][]float64 // the intermediate spin S_t of each path, for diagnostics
}

// NumFuncs is the number of doublet spin functions for this many open shells.
func (b *csfBasis) NumFuncs() int { return len(b.Coef) }

// csfCache memoizes the basis per open-shell count. The counts in play are tiny
// (1, 3 and 5), and the generator is pure, so one map guarded by first use is
// enough; it is built eagerly in init to keep the accessor lock-free.
var csfCache map[int]*csfBasis

func init() {
	csfCache = make(map[int]*csfBasis, 4)
	for _, n := range []int{1, 3, 5} {
		csfCache[n] = buildCSF(n)
	}
}

// doubletCSF returns the doublet spin-function basis for n open shells. n must be
// odd and positive — an even open-shell count cannot carry M_s = +1/2, and every
// ADC configuration of the cation has an odd number of open shells.
func doubletCSF(n int) *csfBasis {
	if b, ok := csfCache[n]; ok {
		return b
	}
	return buildCSF(n)
}

// buildCSF enumerates the M_s = +1/2 determinants and the S = 1/2 branching paths
// for n open shells, and fills the coefficient matrix.
func buildCSF(n int) *csfBasis {
	b := &csfBasis{N: n}
	b.Dets = msDets(n)
	paths := doubletPaths(n)
	b.paths = paths
	b.Coef = make([][]float64, len(paths))
	for p, path := range paths {
		row := make([]float64, len(b.Dets))
		for d, det := range b.Dets {
			row[d] = pathCoef(path, det)
		}
		b.Coef[p] = row
	}
	return b
}

// msDets lists the spin strings of n open shells with exactly one more up than
// down (M_s = +1/2), in ascending lexicographic order of the string (up < down),
// so the determinant order is a pure function of n.
func msDets(n int) [][]int {
	want := (n + 1) / 2 // number of up spins
	if 2*want-n != 1 {
		return nil // n even: no M_s = +1/2 determinant
	}
	var out [][]int
	cur := make([]int, n)
	var rec func(pos, ups int)
	rec = func(pos, ups int) {
		if pos == n {
			if ups == want {
				out = append(out, append([]int(nil), cur...))
			}
			return
		}
		// Prune: neither too many ups already nor too few possible.
		if ups <= want {
			cur[pos] = spinUp
			rec(pos+1, ups+1)
		}
		if pos-ups <= n-want {
			cur[pos] = spinDown
			rec(pos+1, ups)
		}
	}
	rec(0, 0)
	return out
}

// doubletPaths lists the branching-diagram paths S_1..S_n that start from S_0 = 0,
// step by +/- 1/2 without going negative, and end at S_n = 1/2.
//
// Paths are emitted with the DOWN branch first at every step, which fixes the
// spin-function index order. For n = 3 that makes path 0 the one with S_2 = 0 —
// the two leading open shells coupled to a singlet — and path 1 the one with
// S_2 = 1. That ordering is not cosmetic: it is what makes path index 0/1 agree
// with Config.Typ's "spin I"/"spin II" for a 2h1p configuration, whose two holes
// are the leading shells. TestSpinAdaptGateC12_1 pins it.
func doubletPaths(n int) [][]float64 {
	var out [][]float64
	cur := make([]float64, n)
	var walk func(t int, s float64)
	walk = func(t int, s float64) {
		if t == n {
			if s == 0.5 {
				out = append(out, append([]float64(nil), cur...))
			}
			return
		}
		// Unreachable-target pruning: |S - 1/2| must be closable in the
		// remaining n-t steps.
		rem := float64(n - t)
		for _, ns := range []float64{s - 0.5, s + 0.5} {
			if ns < 0 {
				continue
			}
			if math.Abs(ns-0.5) > rem-1 {
				continue
			}
			cur[t] = ns
			walk(t+1, ns)
		}
	}
	walk(0, 0)
	return out
}

// pathCoef is the coefficient of determinant det in the spin function labelled by
// path: the product over shells of the Clebsch-Gordan coefficient coupling the
// running spin S_{t-1} to the shell's own 1/2.
func pathCoef(path []float64, det []int) float64 {
	c := 1.0
	s := 0.0 // S_{t-1}
	m := 0.0 // M_{t-1}
	for t, sp := range det {
		mt := 0.5
		if sp == spinDown {
			mt = -0.5
		}
		c *= cgHalf(s, m, mt, path[t])
		if c == 0 {
			return 0
		}
		s = path[t]
		m += mt
	}
	return c
}

// cgHalf is the Clebsch-Gordan coefficient <j, m; 1/2, mt | jNew, m+mt> for
// coupling a spin-1/2 onto total spin j, with jNew = j +/- 1/2. Standard closed
// forms (Condon-Shortley phase):
//
//	jNew = j + 1/2:  mt = +1/2 -> +sqrt((j + m + 1) / (2j + 1))
//	                 mt = -1/2 -> +sqrt((j - m + 1) / (2j + 1))
//	jNew = j - 1/2:  mt = +1/2 -> -sqrt((j - m) / (2j + 1))
//	                 mt = -1/2 -> +sqrt((j + m) / (2j + 1))
//
// written here in terms of the *initial* m (the tables are usually given in terms
// of the final M = m + mt, which is where sign slips come from).
func cgHalf(j, m, mt, jNew float64) float64 {
	den := 2*j + 1
	switch {
	case jNew > j: // j + 1/2
		if mt > 0 {
			return math.Sqrt((j + m + 1) / den)
		}
		return math.Sqrt((j - m + 1) / den)
	default: // j - 1/2
		if mt > 0 {
			return -math.Sqrt(max0(j-m) / den)
		}
		return math.Sqrt(max0(j+m) / den)
	}
}

// max0 clamps a tiny negative value produced by an unreachable branch to zero, so
// Sqrt never sees it. A genuinely negative argument means the path was invalid,
// and the coefficient is zero either way.
func max0(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}
