package fano

import (
	"math"
	"path/filepath"
	"sync"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
)

const auToMeV = 27211.386245988

// fanoRun is the whole scheme A pipeline over a reduced H2O ADC(2,2) sector, so the F4 and
// F5 gates below each start from a real coupling vector instead of a synthetic one.
type fanoRun struct {
	parent *sip.Space
	pmx    *sip.Matrix
	build  func(*sip.Space) *sip.Matrix
	part   *Partition
	qsp    *sip.Space
	psp    *sip.Space
	phi    Discrete
	g      []float64
	pres   lanczos.Result // the PMP spectrum, solved densely
}

// Memoized per (norb, vacancy): at norb=12 the parent is a 2350x2350 ADC(2,2)f matrix and
// the PMP solve a dense 1180x1180 eigenproblem, ~15 s the pair, and six tests want the same
// one. Nothing here is mutated after construction — no test calls Release, and the matrices
// are read through ApplyFull/BuildMatrix only — so one run serves all of them.
var (
	runMu    sync.Mutex
	runCache = map[[2]int]*fanoRun{}
)

func newFanoRun(t *testing.T, norb, vacancy int) *fanoRun {
	t.Helper()
	runMu.Lock()
	defer runMu.Unlock()
	if r, ok := runCache[[2]int{norb, vacancy}]; ok {
		return r
	}
	r := buildFanoRun(t, norb, vacancy)
	runCache[[2]int{norb, vacancy}] = r
	return r
}

func buildFanoRun(t *testing.T, norb, vacancy int) *fanoRun {
	t.Helper()
	parent, pmx, build := h2o22(t, norb, sip.VariantF)
	sel, err := NewHoleLocalization(AnyHole, []int{vacancy}, "initial vacancy")
	if err != nil {
		t.Fatal(err)
	}
	part := NewPartition(parent, sel)
	if err := part.Validate(); err != nil {
		t.Fatal(err)
	}
	qsp := parent.Restrict(part.Q)
	psp := parent.Restrict(part.P)
	qres := lanczos.SolveDense(build(qsp), backend.Gonum{})
	phi, err := SelectDiscrete(qsp, qres, vacancy, 0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	g, err := Coupling(pmx, part, phi, backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	pres := lanczos.SolveDense(build(psp), backend.Gonum{})
	return &fanoRun{parent, pmx, build, part, qsp, psp, phi, g, pres}
}

// TestSumRuleIsExactOnACompleteBasis is the load-bearing gate of F4.
//
// The chi_i are an orthonormal basis of P, so sum_i <g|chi_i>^2 = ||g||^2 exactly when
// the basis is complete. With SolveDense the Ritz vectors ARE complete, so with every cut
// disabled the discrete widths must add up to 2 pi ||g||^2 to roundoff. That single
// identity ties together the coupling vector, the overlap contraction and the 2 pi, and
// it is the internal check that stands in for the reference tape this layer cannot have:
// adc2_pol's Fano path is commented out of its own build, so no matched matrix-tape gate
// exists, and the reference never sums its widths at all.
func TestSumRuleIsExactOnACompleteBasis(t *testing.T) {
	r := newFanoRun(t, 8, 0)
	res := r.pres
	noCuts := Filter{EMax: -1, MinWeight: -1, MinGamma: -1}
	ps, err := Widths(r.psp, res, r.g, noCuts)
	if err != nil {
		t.Fatal(err)
	}
	if len(ps.Energy) != len(res.Values) {
		t.Errorf("with every cut disabled %d of %d states were dropped",
			len(res.Values)-len(ps.Energy), len(res.Values))
	}
	rel := math.Abs(ps.SumGamma-ps.SumRule) / ps.SumRule
	t.Logf("complete basis over %d P configurations: sum gamma = %.14g, 2pi||g||^2 = %.14g, "+
		"relative gap %.3g", r.psp.Size(), ps.SumGamma, ps.SumRule, rel)
	t.Logf("Gamma = %.6g a.u. = %.4f meV; tau = %.4g fs",
		ps.SumGamma, ps.SumGamma*auToMeV, 658.2119/(ps.SumGamma*auToMeV))
	if rel > 1e-10 {
		t.Errorf("the sum rule is violated by %.3g on a complete basis", rel)
	}
	if ps.DecayHoles != 2 {
		t.Errorf("P's decay class was identified as %dh, want 2h (2h1p) for single ionization",
			ps.DecayHoles)
	}
}

// TestFiltersAccountForEveryState requires the cuts to be exhaustive and their bookkeeping
// to balance: every state is either retained or counted against exactly one cut, and the
// width they carried off is LostGamma. A cut that quietly dropped coupling is the failure
// mode the reference's hard-coded constants hide, since it never checks its own total.
func TestFiltersAccountForEveryState(t *testing.T) {
	r := newFanoRun(t, 8, 0)
	res := r.pres

	exact, err := Widths(r.psp, res, r.g, Filter{EMax: -1, MinWeight: -1, MinGamma: -1})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []Filter{
		{},                                       // the reference's defaults
		{EMax: 1.0, MinWeight: -1, MinGamma: -1}, // energy only
		{EMax: -1, MinWeight: 0.5, MinGamma: -1}, // weight only
		{EMax: -1, MinWeight: -1, MinGamma: 1e-6},
	} {
		ps, err := Widths(r.psp, res, r.g, f)
		if err != nil {
			t.Fatal(err)
		}
		total := len(ps.Energy) + ps.DroppedEnergy + ps.DroppedWeight + ps.DroppedGamma
		if total != len(res.Values) {
			t.Errorf("filter %+v: %d retained + %d/%d/%d dropped = %d, want %d states",
				f, len(ps.Energy), ps.DroppedEnergy, ps.DroppedWeight, ps.DroppedGamma,
				total, len(res.Values))
		}
		if d := math.Abs(ps.SumGamma + ps.LostGamma - exact.SumGamma); d > 1e-14*exact.SumGamma {
			t.Errorf("filter %+v: retained %.6g + lost %.6g != unfiltered %.6g (off by %.3g)",
				f, ps.SumGamma, ps.LostGamma, exact.SumGamma, d)
		}
		if ps.SumRule != exact.SumRule {
			t.Errorf("filter %+v: the sum rule moved with the cuts (%.17g vs %.17g)",
				f, ps.SumRule, exact.SumRule)
		}
		t.Logf("%s", ps)
	}
}

// TestStartRowsIsFillStvc checks the start-block selection against its definition: the k
// rows with the largest g^2, returned ascending, ties broken by index.
func TestStartRowsIsFillStvc(t *testing.T) {
	g := []float64{0.1, -0.9, 0.0, 0.5, 0.9, -0.2}
	// g^2 = 0.01, 0.81, 0, 0.25, 0.81, 0.04 -> descending: rows 1,4 (tie, 1 first), 3, 5, 0, 2
	for _, c := range []struct {
		k    int
		want []int
	}{
		{1, []int{1}},
		{2, []int{1, 4}},
		{3, []int{1, 3, 4}},
		{4, []int{1, 3, 4, 5}},
		{99, []int{0, 1, 2, 3, 4, 5}},
		{0, nil},
	} {
		got := StartRows(g, c.k)
		if len(got) != len(c.want) {
			t.Fatalf("StartRows(k=%d) = %v, want %v", c.k, got, c.want)
		}
		for i := range c.want {
			if got[i] != c.want[i] {
				t.Fatalf("StartRows(k=%d) = %v, want %v", c.k, got, c.want)
			}
		}
	}
}

// TestStartRowsClosesTheSumRuleFaster is why fill_stvc exists. A truncated Krylov space
// captures the width only to the extent its seed overlaps g, so seeding with the most
// strongly coupled configurations must leave a smaller sum-rule residual than seeding
// with the first rows of the P space. This compares the two at the same subspace size.
func TestStartRowsClosesTheSumRuleFaster(t *testing.T) {
	r := newFanoRun(t, 8, 0)
	pmx := r.build(r.psp)
	main := r.psp.MainBlockSize()
	if main < 1 {
		t.Skip("P retained no main-class configuration, so there is no block to seed")
	}
	noCuts := Filter{EMax: -1, MinWeight: -1, MinGamma: -1}

	run := func(rows []int) *Pseudo {
		res := lanczos.Solve(pmx, backend.Gonum{}, lanczos.Options{
			MaxBlocks: 12, WantFull: true, StartRows: rows,
		})
		ps, err := Widths(r.psp, res, r.g, noCuts)
		if err != nil {
			t.Fatal(err)
		}
		return ps
	}
	def := run(nil)
	sel := run(StartRows(r.g, main))
	t.Logf("block width %d, 12 blocks: default seed captures %.4f%% of the sum rule, "+
		"the fill_stvc seed %.4f%%",
		main, 100*(1-def.Residual()), 100*(1-sel.Residual()))
	if sel.Residual() > def.Residual() {
		t.Errorf("seeding with the largest-|g| rows captured LESS of the width "+
			"(residual %.4g) than the default seed (%.4g); fill_stvc's whole purpose is the "+
			"opposite", sel.Residual(), def.Residual())
	}
}

// TestLanczosStartRowsRejectsBadInput pins the contract on the new option: a wrong count,
// an out-of-range row or a repeat each panic rather than silently producing a
// rank-deficient start block, which would shrink the Krylov space without failing.
func TestLanczosStartRowsRejectsBadInput(t *testing.T) {
	r := newFanoRun(t, 8, 0)
	pmx := r.build(r.psp)
	main := r.psp.MainBlockSize()
	cases := []struct {
		name string
		rows []int
	}{
		{"wrong count", fillRowsAscending(main + 1)},
		{"out of range", fillRows(main, r.psp.Size())},
	}
	if main >= 2 {
		// With a block width of 1 a repeated row is just that one row, which is valid.
		cases = append(cases, struct {
			name string
			rows []int
		}{"repeated", fillRows(main, 0)})
	}
	for _, c := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: StartRows was accepted", c.name)
				}
			}()
			lanczos.Solve(pmx, backend.Gonum{}, lanczos.Options{
				MaxBlocks: 2, StartRows: c.rows,
			})
		}()
	}
}

// fillRows builds a length-n slice whose every entry is v (for the repeat and
// out-of-range cases); fillRowsAscending builds a valid-but-wrong-length one.
func fillRows(n, v int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func fillRowsAscending(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

// TestPartialWidthsAreAdditive gates the channel decomposition's defining property: the
// channel widths plus the unassigned remainder reproduce the total exactly. That holds by
// construction for the proportional distribution used here and would NOT hold for the
// alternative reading gamma_beta = 2 pi <g|P_beta|chi>^2, since squaring does not
// distribute over the projector sum — which is why the additive form is the one that can
// express a branching ratio.
func TestPartialWidthsAreAdditive(t *testing.T) {
	r := newFanoRun(t, 8, 0)
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatalf("read MO sidecar: %v", err)
	}
	sites := []spectrum.Site{{Name: "O", Members: []string{"O"}}, {Name: "H", Members: []string{"H1", "H2"}}}
	opts := spectrum.DefaultOptions()
	opts.MinWeight = 0 // keep every channel, so the additivity check is not masked by filtering
	ch, err := NewChannels(md, r.parent.Nocc, sites, "O", opts)
	if err != nil {
		t.Fatal(err)
	}

	res := r.pres
	ps, err := Widths(r.psp, res, r.g, Filter{EMax: -1, MinWeight: -1, MinGamma: -1})
	if err != nil {
		t.Fatal(err)
	}
	pw, err := ch.PartialWidths(r.psp, res, ps)
	if err != nil {
		t.Fatal(err)
	}

	var sum float64
	for _, s := range pw.Sum {
		sum += s
	}
	t.Logf("%s", pw)
	t.Logf("branching ratios: %v over %v", pw.Ratios(), pw.Names)
	if d := math.Abs(sum + pw.Unassigned - ps.SumGamma); d > 1e-12*ps.SumGamma {
		t.Errorf("channels (%.6g) + unassigned (%.6g) != total (%.6g), off by %.3g",
			sum, pw.Unassigned, ps.SumGamma, d)
	}
	// Per state, too — not only in aggregate.
	for i := range ps.Energy {
		var s float64
		for b := range pw.Names {
			s += pw.Gamma[b][i]
		}
		if s > ps.Gamma[i]*(1+1e-12)+1e-300 {
			t.Fatalf("state %d: channels sum to %.6g, above its gamma %.6g", i, s, ps.Gamma[i])
		}
	}
	if len(pw.Names) == 0 {
		t.Error("no decay channels were identified")
	}
	// The second-order channel must be present and last: the 3h2p class is in P.
	if last := pw.Names[len(pw.Names)-1]; last != DoubleName {
		t.Errorf("the last channel is %q, want %q so the AD/CAD split reads in that order",
			last, DoubleName)
	}
}

// TestMullikenPopulationsSumToOne checks the orbital-to-site assignment: a normalized MO's
// gross atomic populations sum to 1, which is what makes the hole spreading in `spread` a
// weight distribution rather than an arbitrary scaling.
func TestMullikenPopulationsSumToOne(t *testing.T) {
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatalf("read MO sidecar: %v", err)
	}
	const nocc = 5
	ch, err := NewChannels(md, nocc, []spectrum.Site{{Name: "O", Members: []string{"O"}}}, "O",
		spectrum.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	for i := range nocc {
		var s float64
		for _, q := range ch.pop[i] {
			s += q
		}
		if math.Abs(s-1) > 1e-10 {
			t.Errorf("orbital %d: gross populations sum to %.12g, want 1", i, s)
		}
	}
	t.Logf("orbital 0 populations over %v: %v", ch.cols, ch.pop[0])
	t.Logf("orbital 4 populations over %v: %v", ch.cols, ch.pop[4])
}

// TestWidthsErrors requires each misuse to be reported rather than producing a plausible
// but wrong width.
func TestWidthsErrors(t *testing.T) {
	r := newFanoRun(t, 8, 0)
	res := r.pres
	if _, err := Widths(r.psp, lanczos.Result{}, r.g, Filter{}); err == nil {
		t.Error("a solve without full Ritz vectors was accepted")
	}
	if _, err := Widths(r.psp, res, r.g[:len(r.g)-1], Filter{}); err == nil {
		t.Error("a coupling vector of the wrong length was accepted")
	}
	if _, err := Widths(r.qsp, res, r.g, Filter{}); err == nil {
		t.Error("a spectrum from a different space was accepted")
	}
	// Coupling must be given the parent, not a restricted matrix.
	if _, err := Coupling(r.build(r.qsp), r.part, r.phi, backend.Gonum{}); err == nil {
		t.Error("Coupling accepted a restricted matrix instead of the parent")
	}
}
