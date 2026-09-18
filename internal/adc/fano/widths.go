package fano

import (
	"fmt"
	"math"

	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// widths.go — turning a solved PMP spectrum into the discrete pseudo-continuum
// {eps_i, gamma_i} that Stieltjes imaging is then applied to.

// Final-state cut defaults. The reference hard-codes all three; here they are the
// defaults of flagged parameters, because none of them is a property of the method:
//
//   - ReferenceEMax is its `etest < 4` a.u. ceiling (fspace.f90:690). It is NOT a default
//     here, and must not be: 4 hartree is a polarization-propagator scale. That code's
//     energies are neutral EXCITATION energies of a few tenths of a hartree, so a 4 Eh
//     ceiling trims a tail. Ionization energies are an order of magnitude larger and a core
//     vacancy is two — Ne 1s sits at 32 Eh — so the same constant silently discards the
//     entire decay continuum and leaves imaging with nothing to interpolate. An energy
//     ceiling is a property of the system, not of the method, so Filter's default is no
//     ceiling at all and the transferable cuts (weight, gamma floor) do the filtering.
//
//   - DefaultMinWeight is its `cntr > 0.05` weight cut (fspace.f90:691; the
//     ADC(2)-extended variant uses 0.2 instead, partgammas.f90:200). TRANSLATE THIS WITH
//     CARE, in two steps.
//
//     First, the reference sums the weight over its own hpdimf configurations, which in
//     the polarization propagator is the 1h1p class — so the naive transfer to ionization
//     would be the 1h weight. It is not: a 1h state is a different ionization, not a decay
//     product. What the cut is really asking is "does this state live in the decay
//     continuum", and for single ionization the decay continuum is everything ABOVE the 1h
//     main class.
//
//     Second, and this is what makes it wrong to read the cut as "the 2h1p weight": at
//     ADC(2,2) the P space has a 3h2p class too, and those states are decay channels —
//     they are the double Auger / double ICD final states, the whole reason the scheme
//     exists. A cut on 2h1p weight alone preferentially discards them, because a
//     3h2p-dominated state has little 2h1p character by construction, and it would remove
//     exactly the second-order decay the method was built to describe. So the weight is
//     summed over EVERY satellite class, which reduces to the reference's reading wherever
//     the reference applies (an order-2/3 P space has only 2h1p above the main block).
//
//   - DefaultMinGamma is its 1e-16 floor (get_gamma_1, fspace.f90:534).
const (
	ReferenceEMax    = 4.0
	DefaultMinWeight = 0.05
	DefaultMinGamma  = 1e-16
)

// Filter holds the final-state cuts. A zero field takes the corresponding default; pass
// a negative value to disable a cut entirely. EMax has no default — zero means no energy
// ceiling — for the reason given above.
type Filter struct {
	EMax      float64 // keep states with eps < EMax (hartree); 0 or negative = no ceiling
	MinWeight float64 // keep states with decay-class weight > MinWeight
	MinGamma  float64 // keep states with gamma > MinGamma (hartree squared)
}

func (f Filter) withDefaults() Filter {
	if f.EMax <= 0 {
		f.EMax = math.Inf(1)
	}
	if f.MinWeight == 0 {
		f.MinWeight = DefaultMinWeight
	}
	if f.MinGamma == 0 {
		f.MinGamma = DefaultMinGamma
	}
	if f.MinWeight < 0 {
		f.MinWeight = math.Inf(-1)
	}
	if f.MinGamma < 0 {
		f.MinGamma = math.Inf(-1)
	}
	return f
}

// Pseudo is the discrete pseudo-continuum: the retained PMP states with their partial
// widths, plus the diagnostics that say how much of the coupling the Krylov space caught.
type Pseudo struct {
	Energy []float64 // eps_i (hartree), ascending
	Gamma  []float64 // gamma_i = 2 pi <g|chi_i>^2 (hartree)
	Weight []float64 // each state's weight on P's decay class, the MinWeight criterion
	Root   []int     // which PMP root each entry came from

	// SumGamma is the total over the RETAINED states and SumRule is 2 pi ||g||^2. They
	// coincide when the Krylov space is complete and nothing is cut, so their gap is this
	// run's discretization error.
	//
	// MIND THE UNITS: these are the zeroth moment of the pseudo-spectrum, not a width.
	// The L2 states chi_i are normalized to 1 rather than to unit energy density, so each
	// gamma_i carries energy SQUARED, and so does their sum. The decay width is the
	// DENSITY dF/dE of that distribution, which is what Stieltjes imaging extracts and
	// what has units of energy. Quoting SumGamma in meV is a category error.
	SumGamma float64
	SumRule  float64

	// DroppedEnergy / DroppedWeight / DroppedGamma count the states each cut removed, in
	// the order the cuts are applied, and LostGamma is the width they carried off. A
	// large LostGamma means a cut is throwing away real coupling, which is the failure
	// mode the reference's hard-coded constants can hide.
	DroppedEnergy, DroppedWeight, DroppedGamma int
	LostGamma                                  float64

	// DecayHoles is the hole count of P's LOWEST decay class (2 for single ionization,
	// 3 for double) and DecayRows the number of rows the weight was summed over — every
	// satellite class, not only the lowest. ClassRows breaks that down by hole count, so a
	// run log shows how much of P is second-order decay.
	DecayHoles, DecayRows int
	ClassRows             map[int]int
}

// Residual is the fraction of the exact width the retained pseudo-continuum misses,
// (SumRule - SumGamma)/SumRule. It is the number to quote as a run's discretization
// error and to gate a fixture on; the reference computes nothing like it, because it
// never sums its widths at all.
func (p *Pseudo) Residual() float64 {
	if p.SumRule == 0 {
		return 0
	}
	return (p.SumRule - p.SumGamma) / p.SumRule
}

// Widths computes the pseudo-continuum from a solved PMP spectrum and the coupling
// vector.
//
//	gamma_i = 2 pi <g|chi_i>^2
//
// psp is the RESTRICTED P space, res its spectrum solved with lanczos.Options.WantFull
// (the satellite components of each chi_i are most of the overlap), and g the coupling
// from Coupling. This is read_gammavec (../ADC/adc2_pol/fspace.f90:644-710) together
// with get_gamma_1 (:534), with its three hard-coded cuts turned into Filter.
//
// Parallel over roots; each root's overlap is accumulated serially over rows, so every
// gamma_i is bit-reproducible whatever the worker count.
func Widths(psp Space, res lanczos.Result, g []float64, f Filter) (*Pseudo, error) {
	f = f.withDefaults()
	if !res.HasFull() {
		return nil, fmt.Errorf("fano: the PMP solve did not retain full Ritz vectors; " +
			"gamma_i is an overlap over every P row, so solve P with lanczos.Options.WantFull")
	}
	n := psp.Size()
	if len(g) != n {
		return nil, fmt.Errorf("fano: the coupling vector has %d components but the P space "+
			"has %d configurations", len(g), n)
	}
	if res.FullVecs.Rows != n {
		return nil, fmt.Errorf("fano: the PMP Ritz vectors have %d rows but the P space has %d",
			res.FullVecs.Rows, n)
	}

	decayHoles, decayRows, classRows := decayClass(psp)
	if len(decayRows) == 0 {
		return nil, fmt.Errorf("fano: the continuum subspace P holds no satellite configuration, "+
			"so it has no decay channels (%d rows, all main class)", n)
	}

	nr := len(res.Values)
	gamma := make([]float64, nr)
	weight := make([]float64, nr)
	parallel.Chunks(nr, parallel.ChunkWorkers(nr), func(_, lo, hi int) {
		for j := lo; j < hi; j++ {
			var ov float64
			for r := range n {
				ov += g[r] * res.FullVecs.At(r, j)
			}
			gamma[j] = 2 * math.Pi * ov * ov
			var w float64
			for _, r := range decayRows {
				c := res.FullVecs.At(r, j)
				w += c * c
			}
			weight[j] = w
		}
	})

	ps := &Pseudo{SumRule: SumRule(g), DecayHoles: decayHoles, DecayRows: len(decayRows),
		ClassRows: classRows}
	// The cuts, in the reference's order and with its comparisons: eps < EMax,
	// weight > MinWeight, gamma > MinGamma.
	for j := range nr {
		switch {
		case !(res.Values[j] < f.EMax):
			ps.DroppedEnergy++
			ps.LostGamma += gamma[j]
		case !(weight[j] > f.MinWeight):
			ps.DroppedWeight++
			ps.LostGamma += gamma[j]
		case !(gamma[j] > f.MinGamma):
			ps.DroppedGamma++
			ps.LostGamma += gamma[j]
		default:
			ps.Energy = append(ps.Energy, res.Values[j])
			ps.Gamma = append(ps.Gamma, gamma[j])
			ps.Weight = append(ps.Weight, weight[j])
			ps.Root = append(ps.Root, j)
			ps.SumGamma += gamma[j]
		}
	}
	return ps, nil
}

// decayClass returns P's decay-continuum rows — every configuration above the main block,
// whatever its class — together with the hole count of the lowest such class and a
// per-class row census.
//
// Every satellite class, for the reason given at DefaultMinWeight: at ADC(2,2) the 3h2p
// configurations are the second-order decay channels, so restricting the weight to the
// lowest class would discard them.
//
// The lowest class is reported as the MINIMUM hole count above the main block rather than
// "main + 1", so it is right even for a P space that retained no main-class configuration
// at all — which happens whenever the Q criterion claims every main-class row, and would
// otherwise make the class boundary unknowable from P alone.
func decayClass(sp Space) (holes int, rows []int, census map[int]int) {
	main := sp.MainBlockSize()
	buf := make([]int, 0, 4)
	holes = -1
	census = map[int]int{}
	for r := main; r < sp.Size(); r++ {
		h := len(sp.Holes(r, buf[:0]))
		census[h]++
		if holes < 0 || h < holes {
			holes = h
		}
		rows = append(rows, r)
	}
	if holes < 0 {
		return 0, nil, census
	}
	return holes, rows, census
}

// String summarizes the pseudo-continuum for the run log.
func (p *Pseudo) String() string {
	classes := ""
	for _, h := range []int{2, 3, 4} {
		if n := p.ClassRows[h]; n > 0 {
			classes += fmt.Sprintf(" %dh:%d", h, n)
		}
	}
	return fmt.Sprintf("%d pseudo-continuum states (decay continuum%s of P, %d rows total); "+
		"sum gamma = %.6g Eh^2, sum rule 2pi||g||^2 = %.6g Eh^2, residual %.3g%%; "+
		"dropped %d by energy, %d by weight, %d by gamma (carrying %.3g Eh^2)",
		len(p.Energy), classes, p.DecayRows,
		p.SumGamma, p.SumRule, 100*p.Residual(),
		p.DroppedEnergy, p.DroppedWeight, p.DroppedGamma, p.LostGamma)
}
