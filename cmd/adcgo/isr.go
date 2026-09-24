package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	isrdip "github.com/leiaSQ/ADCgo/internal/adc/isrgen/dip"
	isrip "github.com/leiaSQ/ADCgo/internal/adc/isrgen/ip"
	isrqip "github.com/leiaSQ/ADCgo/internal/adc/isrgen/qip"
	"github.com/leiaSQ/ADCgo/internal/adc/isrgen/sigma"
	isrtip "github.com/leiaSQ/ADCgo/internal/adc/isrgen/tip"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// isr.go — -isr VARIANT:SCHEME: the spectrum of a generated ISR secular matrix
// (internal/adc/isrgen), applied as tensor contractions (internal/adc/isrgen/sigma) on a
// khci space of one Ms sector. Each multiplicity is its own sector: block Lanczos from
// the pure-spin main-class vectors, every apply projected onto that spin.

const isrAu2eV = 27.211396 // the conversion analyze reports -dip/-sip energies with

// isrVariant is a generated package with a σ program.
type isrVariant struct {
	k        int
	schemes  map[string][6]int
	newSigma func(sp *khci.Space, ints *integrals.Store, eps []float64, nocc int, scheme string,
		be backend.Backend) (*sigma.Operator, error)
}

var isrVariants = map[string]isrVariant{
	"ip":  {isrip.K, isrip.SigmaSchemes, isrip.NewSigma},
	"dip": {isrdip.K, isrdip.SigmaSchemes, isrdip.NewSigma},
	"tip": {isrtip.K, isrtip.SigmaSchemes, isrtip.NewSigma},
	"qip": {isrqip.K, isrqip.SigmaSchemes, isrqip.NewSigma},
}

type isrConfig struct {
	variant, scheme       string
	solver, spinSel, sym  string
	backend, out          string
	maxClass, twoMs       int // khci space; 0 / -1 = the scheme's classes / the lowest Ms
	mgpu                  int // > 1: split every apply across this many devices (column split)
	blocks                int
	psThresh, coeffThresh float64
	profile               bool
}

// parseISR splits -isr VARIANT:SCHEME and checks both exist.
func parseISR(s string) (string, string, error) {
	v, sc, ok := strings.Cut(s, ":")
	if !ok {
		return "", "", fmt.Errorf("-isr %q: want VARIANT:SCHEME, e.g. dip:adc2x", s)
	}
	iv, ok := isrVariants[v]
	if !ok {
		var have []string
		for n := range isrVariants {
			have = append(have, n)
		}
		sort.Strings(have)
		return "", "", fmt.Errorf("-isr: no σ program for variant %q (have %v)", v, have)
	}
	if _, ok := iv.schemes[sc]; !ok {
		var have []string
		for n := range iv.schemes {
			have = append(have, n)
		}
		sort.Strings(have)
		return "", "", fmt.Errorf("-isr: variant %s has no scheme %q (have %v)", v, sc, have)
	}
	return v, sc, nil
}

// schemeMaxClass is the highest class (as a hole count) any block of the scheme touches.
func schemeMaxClass(k int, mo [6]int) int {
	pairs := [6][2]int{{0, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {2, 2}}
	top := 0
	for b, o := range mo {
		if o >= 0 {
			top = max(top, pairs[b][1])
		}
	}
	return k + top
}

// isrMultiplicities maps -spin to the 2S values to solve: every multiplicity the main
// class holds for "both"/"all", else the named ones.
func isrMultiplicities(sel string, k, twoMs int) ([]int, error) {
	names := map[string]int{"singlet": 0, "doublet": 1, "triplet": 2, "quartet": 3, "quintet": 4}
	if sel == "both" || sel == "all" {
		var out []int
		for s := abs(twoMs); s <= k; s += 2 {
			out = append(out, s)
		}
		return out, nil
	}
	var out []int
	for _, f := range strings.Split(sel, ",") {
		s, ok := names[strings.TrimSpace(f)]
		if !ok {
			return nil, fmt.Errorf("-spin %q: want both|all or a list of singlet..quintet", sel)
		}
		if s > k || (s-twoMs)%2 != 0 || s < abs(twoMs) {
			return nil, fmt.Errorf("-spin %s: 2S=%d is not reachable with %d holes at 2Ms=%d", f, s, k, twoMs)
		}
		out = append(out, s)
	}
	return out, nil
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

type isrLeading struct {
	Holes string  `json:"holes"` // spin orbitals as spatial index and a (alpha) / b (beta), 0-based
	Coeff float64 `json:"coeff"`
}

type isrState struct {
	Index     int          `json:"index"`
	EnergyEV  float64      `json:"energy_ev"`
	PSPercent float64      `json:"ps_percent"`
	Residue   float64      `json:"residue"` // eV; 0 on the dense path
	Leading   []isrLeading `json:"leading"`
}

type isrSector struct {
	Irrep        int        `json:"irrep"` // 1-based
	Multiplicity int        `json:"multiplicity"`
	Size         int        `json:"size"` // rows of the Ms sector the operator acts on
	States       []isrState `json:"states"`
}

type isrDocument struct {
	NORB     int         `json:"norb"`
	NELEC    int         `json:"nelec"`
	Variant  string      `json:"variant"`
	Scheme   string      `json:"scheme"`
	TwoMs    int         `json:"two_ms"`
	MaxClass int         `json:"max_class"`
	Solver   string      `json:"solver"`
	Sectors  []isrSector `json:"sectors"`
}

func runISR(d *fcidump.Data, cfg isrConfig) error {
	iv := isrVariants[cfg.variant]
	nocc := mp.NOcc(d)
	if err := mp.RequireCanonical(d, nocc); err != nil {
		return err
	}
	if cfg.solver != "lanczos" && cfg.solver != "dense" {
		return fmt.Errorf("-isr runs -solver lanczos or dense, not %q", cfg.solver)
	}
	eps := mp.OrbitalEnergies(d, nocc)
	orbSym, syms, err := selectSymmetry(cfg.sym, d)
	if err != nil {
		return err
	}
	ints := integrals.New(d, nocc, orbSym)
	maxClass := cfg.maxClass
	if maxClass == 0 {
		maxClass = schemeMaxClass(iv.k, iv.schemes[cfg.scheme])
	}
	twoMs := cfg.twoMs
	if twoMs < 0 {
		twoMs = iv.k % 2
	}
	spins, err := isrMultiplicities(cfg.spinSel, iv.k, twoMs)
	if err != nil {
		return err
	}
	be, err := backend.New(cfg.backend)
	if err != nil {
		return err
	}
	var subs []backend.Backend
	if cfg.mgpu > 1 {
		if subs, err = backend.NewAll(cfg.backend, cfg.mgpu); err != nil {
			return err
		}
		if len(subs) < cfg.mgpu {
			fmt.Fprintf(os.Stderr, "adcgo: -isr -mgpu %d: %d device(s) visible\n", cfg.mgpu, len(subs))
		}
	}
	doc := isrDocument{NORB: d.NORB, NELEC: d.NELEC, Variant: cfg.variant, Scheme: cfg.scheme,
		TwoMs: twoMs, MaxClass: maxClass, Solver: cfg.solver}
	for _, irrep := range syms {
		opts := khci.Options{K: iv.k, NOcc: nocc, NVir: d.NORB - nocc, TwoMs: twoMs,
			TargetIrrep: irrep, MaxClass: maxClass}
		if orbSym != nil {
			opts.OrbSym = make([]int, d.NORB)
			for p, g := range orbSym {
				opts.OrbSym[p] = g - 1
			}
		}
		sp, err := khci.NewSpace(opts)
		if err != nil {
			return err
		}
		if sp.MainBlockSize() == 0 {
			continue
		}
		sbe := be
		if len(subs) > 1 && sp.Size() > 2*sp.MainBlockSize()*sp.MainBlockSize() {
			// every partition runs the whole program on its share of the panel columns;
			// the row bounds only say where the Krylov vectors live
			n, np := sp.Size(), len(subs)
			bounds := make([]int, np+1)
			for i := range bounds {
				bounds[i] = i * n / np
			}
			if sbe, err = backend.NewDistributed(subs, n, sp.MainBlockSize(), bounds); err != nil {
				return err
			}
		}
		op, err := iv.newSigma(sp, ints, eps, nocc, cfg.scheme, sbe)
		if err != nil {
			return err
		}
		if op.Staged() {
			fmt.Fprintf(os.Stderr, "adcgo: -isr: backend %s has no tensor kernels; every apply stages through the host\n", cfg.backend)
		}
		for _, twoS := range spins {
			sec, ok, err := solveISRSector(op, sp, twoS, cfg, sbe)
			if err != nil {
				return fmt.Errorf("irrep %d, 2S=%d: %w", irrep, twoS, err)
			}
			if !ok {
				continue
			}
			sec.Irrep = irrep + 1
			doc.Sectors = append(doc.Sectors, sec)
		}
		op.Release()
	}
	return emitJSON(doc, cfg.out)
}

// solveISRSector solves one multiplicity; ok is false when the main class holds none.
func solveISRSector(op *sigma.Operator, sp *khci.Space, twoS int, cfg isrConfig, be backend.Backend) (isrSector, bool, error) {
	vecs, b, err := sp.MainSpinVectors(twoS)
	if err != nil || b == 0 {
		return isrSector{}, false, err
	}
	if err := op.SetSpin(twoS); err != nil {
		return isrSector{}, false, err
	}
	var res lanczos.Result
	switch cfg.solver {
	case "lanczos":
		res = lanczos.Solve(op, be, lanczos.Options{Block: b, StartVecs: vecs, MaxBlocks: cfg.blocks})
		if cfg.profile {
			reportTiming(fmt.Sprintf("isr 2S=%d", twoS), sp.Size(), b, res.Timing)
		}
	case "dense":
		if res, err = denseISR(op, sp, twoS, be); err != nil {
			return isrSector{}, false, err
		}
	}
	sec := isrSector{Multiplicity: twoS + 1, Size: sp.Size()}
	main := sp.MainBlockSize()
	for k, e := range res.Values {
		if res.PS[k] < cfg.psThresh || res.Spurious(k, 1e-9) {
			continue
		}
		st := isrState{Index: len(sec.States) + 1, EnergyEV: e * isrAu2eV, PSPercent: res.PS[k]}
		if res.Residual != nil {
			st.Residue = res.Residual[k] * isrAu2eV
		}
		for r := range main {
			if c := res.MainVecs.At(r, k); math.Abs(c) >= cfg.coeffThresh {
				st.Leading = append(st.Leading, isrLeading{Holes: holeLabel(sp, r), Coeff: c})
			}
		}
		sort.Slice(st.Leading, func(i, j int) bool {
			return math.Abs(st.Leading[i].Coeff) > math.Abs(st.Leading[j].Coeff)
		})
		sec.States = append(sec.States, st)
	}
	return sec, true, nil
}

// holeLabel names a main-class row by its holes, e.g. "3a 3b 4a".
func holeLabel(sp *khci.Space, r int) string {
	var parts []string
	for q, p := sp.HoleMask(r), 0; q != 0; q, p = q>>1, p+1 {
		if q&1 == 0 {
			continue
		}
		s := "a"
		if p&1 == 1 {
			s = "b"
		}
		parts = append(parts, fmt.Sprintf("%d%s", p>>1, s))
	}
	return strings.Join(parts, " ")
}

// denseISR diagonalizes P·M·P built column by column from the σ operator and keeps the
// eigenvectors inside the multiplicity (the rest span P's null space at eigenvalue 0).
func denseISR(op *sigma.Operator, sp *khci.Space, twoS int, be backend.Backend) (lanczos.Result, error) {
	n, main := sp.Size(), sp.MainBlockSize()
	proj, err := sp.SpinProjector(twoS)
	if err != nil {
		return lanczos.Result{}, err
	}
	m := backend.NewMat(n, n)
	const chunk = 256
	for c0 := 0; c0 < n; c0 += chunk {
		nc := min(chunk, n-c0)
		x := make([]float64, n*nc)
		for j := range nc {
			x[j*n+c0+j] = 1
		}
		out := be.Alloc(n * nc)
		op.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: nc, Ld: n},
			backend.BlockView{V: be.Upload(x), Rows: n, Cols: nc, Ld: n})
		y := be.Download(out)
		be.Free(out)
		for j := range nc {
			for r := range n {
				m.Data[r*n+c0+j] = y[j*n+r]
			}
		}
	}
	vals, vecs := backend.Gonum{}.SymEig(m)
	var res lanczos.Result
	var keep []int
	col := make([]float64, n)
	for k := range vals {
		for r := range n {
			col[r] = vecs.At(r, k)
		}
		proj.Apply(col, 1, n)
		var w float64
		for _, v := range col {
			w += v * v
		}
		if w > 0.5 {
			keep = append(keep, k)
		}
	}
	res.MainVecs = backend.NewMat(main, len(keep))
	for j, k := range keep {
		res.Values = append(res.Values, vals[k])
		var ps float64
		for r := range main {
			v := vecs.At(r, k)
			res.MainVecs.Data[r*len(keep)+j] = v
			ps += v * v
		}
		res.PS = append(res.PS, 100*ps)
	}
	return res, nil
}
