package sip

import (
	"fmt"
	"math"
	"math/bits"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// ---------------------------------------------------------------------------
// A determinant oracle.
//
// isrdipole.go states the spin functions of the SIP configurations and derives D
// from them in closed form. Everything below re-derives, independently and from
// second quantization alone, what those configurations imply — first for the
// Hamiltonian (which pins the spin functions against the already-validated secular
// matrix elements, phases included) and then for a one-particle operator (which is
// the thing D is supposed to be). Nothing here shares a line of code with the
// production formulas: it expands each configuration into its ≤3 Slater determinants
// and applies the Slater–Condon rules with phases obtained by literally walking the
// creation/annihilation operators through the occupation string.
// ---------------------------------------------------------------------------

// sdet is a Slater determinant over ≤64 spatial orbitals: one occupation bitmask per
// spin. The canonical operator order is every α spin-orbital in ascending orbital
// index, then every β one — so the sign of c_pσ is set by the occupation below it.
type sdet struct{ a, b uint64 }

const (
	alpha = 0
	beta  = 1
)

// occBefore counts the occupied spin-orbitals preceding (p, spin) in canonical order.
func (d sdet) occBefore(p, spin int) int {
	below := bits.OnesCount64(d.a & (uint64(1)<<p - 1))
	if spin == alpha {
		return below
	}
	return bits.OnesCount64(d.a) + bits.OnesCount64(d.b&(uint64(1)<<p-1))
}

// op is one second-quantized operator; strings are applied rightmost-first.
type op struct {
	create bool
	p      int
	spin   int
}

func an(p, spin int) op { return op{false, p, spin} }
func cr(p, spin int) op { return op{true, p, spin} }

// apply walks ops (in application order) across d, returning the resulting determinant
// and its phase, or phase 0 if the string annihilates d.
func apply(d sdet, ops ...op) (sdet, float64) {
	sign := 1.0
	for _, o := range ops {
		m := uint64(1) << o.p
		occ := d.a
		if o.spin == beta {
			occ = d.b
		}
		if (occ&m != 0) == o.create {
			return d, 0 // creating into an occupied, or annihilating an empty, spinorbital
		}
		if d.occBefore(o.p, o.spin)%2 == 1 {
			sign = -sign
		}
		if o.spin == alpha {
			d.a ^= m
		} else {
			d.b ^= m
		}
	}
	return d, sign
}

// dterm is one determinant of a configuration's expansion.
type dterm struct {
	c float64
	d sdet
}

// opTerm is one term of a configuration's spin function: a coefficient and the
// second-quantized operator string that produces it.
type opTerm struct {
	c   float64
	ops []op
}

// configDets expands configuration idx of sp into determinants, using exactly the spin
// functions documented in isrdipole.go — that is, it applies configOps to the reference.
func configDets(sp *Space, idx int) []dterm {
	ref := sdet{a: uint64(1)<<sp.Nocc - 1, b: uint64(1)<<sp.Nocc - 1}
	var out []dterm
	for _, t := range configOps(sp, idx) {
		d, s := apply(ref, t.ops...)
		if s == 0 {
			panic("configDets: operator string annihilates the reference")
		}
		out = append(out, dterm{t.c * s, d})
	}
	return out
}

// configOps gives configuration idx's spin function as operator strings, detached from any
// particular determinant. configDets applies them to the Hartree–Fock reference; the ISR
// oracle (isrdipole_corr_test.go) applies the very same strings to a *correlated* ground
// state, which is the whole difference between a configuration and an intermediate state.
func configOps(sp *Space, idx int) []opTerm {
	push := func(out []opTerm, c float64, ops ...op) []opTerm {
		return append(out, opTerm{c, ops})
	}
	cfg := sp.Configs[idx]
	if idx < sp.BeginSat { // |i> = −c_iβ |0>
		// The leading minus is not a choice. The 1h configurations' phase relative to the
		// 2h1p ones is an observable of the secular matrix — it is the sign of the whole
		// coupling block — and calc_c12_1.c fixes it to this. Drop it and
		// TestSpinFunctionsReproduceSecularBlocks fails on every c12_1 element by exactly
		// a factor of −1.
		return push(nil, -1, an(cfg.Occ[0], beta))
	}
	k, l, a := cfg.Occ[0], cfg.Occ[1], sp.Nocc+cfg.Vir
	if k == l { // |akk> = c†_aα c_kα c_kβ |0>
		return push(nil, 1, an(k, beta), an(k, alpha), cr(a, alpha))
	}
	if cfg.Typ == 0 { // |akl;I> = (1/√2)(c†_aα c_kα c_lβ − c†_aα c_kβ c_lα)|0>
		out := push(nil, sqrt1_2, an(l, beta), an(k, alpha), cr(a, alpha))
		return push(out, -sqrt1_2, an(l, alpha), an(k, beta), cr(a, alpha))
	}
	// |akl;II> = √(2/3) c†_aβ c_kβ c_lβ |0> + (1/√6)(c†_aα c_kα c_lβ + c†_aα c_kβ c_lα)|0>
	//
	// ...except in a CVS ADC(4) space, whose elements (elements4.go) are transcribed from
	// the reference's Dyson-convention kernels, in which the type-II function carries the
	// opposite sign. That is a hypothesis about elements4.go, not a free choice:
	// TestADC4SpinFunctionsReproduceKopp1 fails by exactly −1 on every type-II element if
	// it is dropped, and TestSpinFunctionsReproduceSecularBlocks fails if it is applied to
	// a plain space.
	s := 1.0
	if sp.adc4 {
		s = -1
	}
	out := push(nil, s*math.Sqrt(2.0/3.0), an(l, beta), an(k, beta), cr(a, beta))
	out = push(out, s*math.Sqrt(1.0/6.0), an(l, beta), an(k, alpha), cr(a, alpha))
	return push(out, s*math.Sqrt(1.0/6.0), an(l, alpha), an(k, beta), cr(a, alpha))
}

// so is a spin-orbital.
type so struct{ p, spin int }

// occupied lists the occupied spin-orbitals of d in canonical order.
func (d sdet) occupied() []so {
	var out []so
	for m := d.a; m != 0; m &= m - 1 {
		out = append(out, so{bits.TrailingZeros64(m), alpha})
	}
	for m := d.b; m != 0; m &= m - 1 {
		out = append(out, so{bits.TrailingZeros64(m), beta})
	}
	return out
}

// exclusive lists the spin-orbitals occupied in d but not in e, canonical order.
func exclusive(d, e sdet) []so {
	return sdet{a: d.a &^ e.a, b: d.b &^ e.b}.occupied()
}

// oneBodyDet is <bra| Σ_pq d_pq Σ_σ c†_pσ c_qσ |ket>.
func oneBodyDet(bra, ket sdet, d backend.Mat) float64 {
	bo, ko := exclusive(bra, ket), exclusive(ket, bra)
	switch {
	case len(bo) == 0:
		var acc float64
		for _, s := range ket.occupied() {
			acc += d.At(s.p, s.p)
		}
		return acc
	case len(bo) == 1 && bo[0].spin == ko[0].spin:
		p, q, s := bo[0].p, ko[0].p, bo[0].spin
		if _, g := apply(ket, an(q, s), cr(p, s)); g != 0 {
			return g * d.At(p, q)
		}
		return 0
	default:
		return 0
	}
}

// phys is the physicist integral <tu|vw> over spatial orbitals, = (tv|uw) in chemist
// notation, which is what fcidump stores.
func phys(fd *fcidump.Data, t, u, v, w int) float64 { return fd.TwoE(t, v, u, w) }

// hamDet is <bra|Ĥ|ket> for the electronic Hamiltonian (no nuclear repulsion), by the
// Slater–Condon rules. Phases come from the operator strings, not from a counting
// formula, so there is nothing here to get subtly backwards.
func hamDet(bra, ket sdet, fd *fcidump.Data) float64 {
	bo, ko := exclusive(bra, ket), exclusive(ket, bra)
	switch len(bo) {
	case 0:
		occ := ket.occupied()
		var acc float64
		for _, P := range occ {
			acc += fd.OneE(P.p, P.p)
		}
		for _, P := range occ {
			for _, Q := range occ {
				acc += 0.5 * phys(fd, P.p, Q.p, P.p, Q.p)
				if P.spin == Q.spin {
					acc -= 0.5 * phys(fd, P.p, Q.p, Q.p, P.p)
				}
			}
		}
		return acc

	case 1:
		P, Q := bo[0], ko[0]
		if P.spin != Q.spin {
			return 0
		}
		_, g := apply(ket, an(Q.p, Q.spin), cr(P.p, P.spin))
		if g == 0 {
			return 0
		}
		acc := fd.OneE(P.p, Q.p)
		for _, R := range ket.occupied() {
			if R == Q {
				continue // R runs over the orbitals common to both determinants
			}
			acc += phys(fd, P.p, R.p, Q.p, R.p)
			if P.spin == R.spin {
				acc -= phys(fd, P.p, R.p, R.p, Q.p)
			}
		}
		return g * acc

	case 2:
		P1, P2, Q1, Q2 := bo[0], bo[1], ko[0], ko[1]
		_, g := apply(ket, an(Q1.p, Q1.spin), an(Q2.p, Q2.spin), cr(P2.p, P2.spin), cr(P1.p, P1.spin))
		if g == 0 {
			return 0
		}
		var acc float64
		if P1.spin == Q1.spin && P2.spin == Q2.spin {
			acc += phys(fd, P1.p, P2.p, Q1.p, Q2.p)
		}
		if P1.spin == Q2.spin && P2.spin == Q1.spin {
			acc -= phys(fd, P1.p, P2.p, Q2.p, Q1.p)
		}
		return g * acc

	default:
		return 0
	}
}

// configME contracts a determinant-level matrix element up to the configuration basis.
func configME(bra, ket []dterm, me func(x, y sdet) float64) float64 {
	var acc float64
	for _, u := range bra {
		for _, v := range ket {
			acc += u.c * v.c * me(u.d, v.d)
		}
	}
	return acc
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

func h2oData(t *testing.T) *fcidump.Data {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	return d
}

// dipoleSpace builds the symmetry-off H2O space (every configuration present, so the
// tests cover every branch of satsat) together with its matrix engine.
func dipoleSpace(t *testing.T) (*Space, *Matrix, *fcidump.Data) {
	t.Helper()
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	sp := NewSpace(nocc, fd.NORB, nil, 0)
	mx := New(sp, integrals.New(fd, nocc, nil), eps, 2, backend.Gonum{})
	return sp, mx, fd
}

// randomSymmetric is a deterministic pseudo-random symmetric operator. It exercises
// every block of d — occupied/occupied, occupied/virtual, virtual/virtual — which a
// real dipole in a symmetric molecule does not.
func randomSymmetric(n int) backend.Mat {
	m := backend.NewMat(n, n)
	state := uint64(0x9E3779B97F4A7C15)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(int64(state>>11))/float64(int64(1)<<52) - 1
	}
	for i := range n {
		for j := 0; j <= i; j++ {
			v := next()
			m.Set(i, j, v)
			m.Set(j, i, v)
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

// TestSpinFunctionsReproduceSecularBlocks is the convention gate the rest of the file
// rests on. The eigenvectors X that D gets contracted with are expressed in whatever
// spin functions elements.go assumed; if isrdipole.go assumes different ones (or the
// same ones with a flipped phase on type II, or with k and l exchanged), every
// transition moment is wrong while every eigenvalue stays right.
//
// So: expand the configurations into determinants using the spin functions isrdipole.go
// documents, evaluate <Φ_I|Ĥ|Φ_J> − E_HF·δ_IJ by Slater–Condon, and require it to
// reproduce the zeroth- plus first-order secular matrix — −ε_i on the 1h diagonal,
// c12_1 on the coupling block, c22diag/c22off on the satellite block. Those blocks are
// exact at that order (the Fock operator is diagonal in this basis, so nothing else can
// contribute), and they are already validated bit-exactly against theADCcode.
func TestSpinFunctionsReproduceSecularBlocks(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping determinant-level secular-matrix check in -short mode")
	}
	sp, mx, fd := dipoleSpace(t)
	eps := mp.OrbitalEnergies(fd, mp.NOcc(fd))
	n := sp.Size()

	dets := make([][]dterm, n)
	for i := range n {
		dets[i] = configDets(sp, i)
	}
	ref := sdet{a: uint64(1)<<sp.Nocc - 1, b: uint64(1)<<sp.Nocc - 1}
	eHF := hamDet(ref, ref, fd)

	me := func(x, y sdet) float64 { return hamDet(x, y, fd) }
	var maxErr float64
	worst := ""
	check := func(i, j int, want float64) {
		got := configME(dets[i], dets[j], me)
		if i == j {
			got -= eHF
		}
		if e := math.Abs(got - want); e > maxErr {
			maxErr = e
			worst = fmt.Sprintf("(%d,%d): configs %v, %v", i, j, sp.Configs[i], sp.Configs[j])
		}
	}
	for i := range sp.BeginSat {
		for j := range sp.BeginSat {
			want := 0.0
			if i == j {
				want = -eps[sp.Configs[i].Occ[0]]
			}
			check(i, j, want)
		}
		for j := sp.BeginSat; j < n; j++ {
			check(i, j, mx.el.c12_1(sp.Configs[i].Occ[0], sp.Configs[j]))
		}
	}
	for i := sp.BeginSat; i < n; i++ {
		check(i, i, mx.el.c22diag(sp.Configs[i]))
		for j := i + 1; j < n; j++ {
			// satBlock's convention: the reference fills the column of the higher index.
			check(i, j, mx.el.c22off(sp.Configs[i], sp.Configs[j]))
		}
	}
	if maxErr > 1e-9 {
		t.Errorf("determinant Hamiltonian departs from the secular blocks by %g at %s;\n"+
			"the spin functions in isrdipole.go are not the ones elements.go uses", maxErr, worst)
	}
}

// TestISRDipoleMatchesDeterminants is the correctness gate on D itself: every element,
// against the same determinant expansion, for an operator whose every block is filled.
func TestISRDipoleMatchesDeterminants(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping determinant-level dipole check in -short mode")
	}
	sp, _, _ := dipoleSpace(t)
	d := randomSymmetric(sp.Norb)
	o, err := NewISRDipole(sp, d)
	if err != nil {
		t.Fatal(err)
	}
	n := sp.Size()
	dets := make([][]dterm, n)
	for i := range n {
		dets[i] = configDets(sp, i)
	}
	me := func(x, y sdet) float64 { return oneBodyDet(x, y, d) }
	var maxErr float64
	for i := range n {
		for j := range n {
			want := configME(dets[i], dets[j], me)
			if e := math.Abs(o.At(i, j) - want); e > maxErr {
				maxErr = e
				if e > 1e-10 {
					t.Fatalf("D[%d,%d] = %g, determinants give %g (configs %v, %v)",
						i, j, o.At(i, j), want, sp.Configs[i], sp.Configs[j])
				}
			}
		}
	}
	if maxErr > 1e-10 {
		t.Errorf("max |D − determinant D| = %g", maxErr)
	}
}

// TestISRDipoleNumberOperator: with d = 1, D̂ is the electron-number operator, so every
// configuration is an eigenvector with eigenvalue N−1. This catches a wrong D0, a
// dropped hole term, and any spurious off-diagonal coupling — without reference to the
// determinant machinery at all.
func TestISRDipoleNumberOperator(t *testing.T) {
	sp, _, _ := dipoleSpace(t)
	id := backend.NewMat(sp.Norb, sp.Norb)
	for p := range sp.Norb {
		id.Set(p, p, 1)
	}
	o, err := NewISRDipole(sp, id)
	if err != nil {
		t.Fatal(err)
	}
	want := float64(2*sp.Nocc - 1)
	for i := range sp.Size() {
		for j := range sp.Size() {
			exp := 0.0
			if i == j {
				exp = want
			}
			if e := math.Abs(o.At(i, j) - exp); e > 1e-12 {
				t.Fatalf("number operator: D[%d,%d] = %g, want %g", i, j, o.At(i, j), exp)
			}
		}
	}
}

// TestISRDipoleApplyMatchesDense: the sparse row structure reaches every nonzero.
func TestISRDipoleApplyMatchesDense(t *testing.T) {
	sp, _, _ := dipoleSpace(t)
	o, err := NewISRDipole(sp, randomSymmetric(sp.Norb))
	if err != nil {
		t.Fatal(err)
	}
	n := o.Size()
	m := o.BuildMatrix()
	in := make([]float64, n)
	out := make([]float64, n)
	var maxErr float64
	for j := range n {
		in[j] = 1
		o.Apply(out, in)
		in[j] = 0
		for i := range n {
			if e := math.Abs(out[i] - m.At(i, j)); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 1e-14 {
		t.Errorf("Apply vs BuildMatrix max diff %g: sparsify missed a nonzero column", maxErr)
	}
}

// TestISRDipoleSymmetric: D is the representation of a symmetric operator.
func TestISRDipoleSymmetric(t *testing.T) {
	sp, _, _ := dipoleSpace(t)
	o, err := NewISRDipole(sp, randomSymmetric(sp.Norb))
	if err != nil {
		t.Fatal(err)
	}
	for i := range o.Size() {
		for j := range i {
			if e := math.Abs(o.At(i, j) - o.At(j, i)); e > 1e-14 {
				t.Fatalf("D asymmetric at (%d,%d): %g vs %g", i, j, o.At(i, j), o.At(j, i))
			}
		}
	}
}

// TestSymmetryForbiddenComponentsVanish: with point-group symmetry on, only the totally
// symmetric Cartesian component of the dipole has matrix elements inside one target
// irrep. For H2O in C2v that is z alone — μ_x and μ_y carry a non-trivial irrep and
// connect a sector to a *different* one, so their in-sector D vanishes identically.
// Nothing in isrdipole.go knows about irreps; the zeros come out of the configuration
// enumeration, which is the point.
func TestSymmetryForbiddenComponentsVanish(t *testing.T) {
	fd := h2oData(t)
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	if !md.HasDipole {
		t.Fatal("h2o.mo.json has no dipole keys")
	}
	if md.NMO != fd.NORB {
		t.Fatalf("sidecar has %d MOs, fcidump %d", md.NMO, fd.NORB)
	}
	nocc := mp.NOcc(fd)
	maxAbs := func(o *ISRDipole) float64 {
		var m float64
		for i := range o.Size() {
			for _, e := range o.rows[i] {
				if a := math.Abs(e.val); a > m {
					m = a
				}
			}
		}
		return m
	}
	sawZ := false
	for sym := range 4 {
		sp := NewSpace(nocc, fd.NORB, fd.OrbSym, sym)
		if sp.MainBlockSize() == 0 {
			continue
		}
		ops, err := NewISRDipoles(sp, md.DipMO)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range []int{0, 1} {
			if m := maxAbs(ops[c]); m > 1e-12 {
				t.Errorf("irrep %d: component %d should be symmetry-forbidden, max |D| = %g",
					sym, c, m)
			}
		}
		if maxAbs(ops[2]) > 0.1 {
			sawZ = true
		}
	}
	if !sawZ {
		t.Error("no sector had an appreciable z-component: the test proved nothing")
	}
}

// TestEmissionsMatchDenseContraction runs the whole chain the way a caller will — a
// dense SIP solve whose eigenvectors carry their satellite rows, contracted through D —
// and checks μ against an explicit Xᵀ·D·X built from the dense matrices.
func TestEmissionsMatchDenseContraction(t *testing.T) {
	sp, mx, _ := dipoleSpace(t)
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	ops, err := NewISRDipoles(sp, md.DipMO)
	if err != nil {
		t.Fatal(err)
	}
	res := lanczos.SolveDense(mx, backend.Gonum{})
	if !res.HasFull() {
		t.Fatal("SolveDense did not retain FullVecs")
	}
	if _, err := Emissions(ops, res.Values, res.MainVecs, []int{1}, []int{0}); err == nil {
		t.Error("Emissions accepted main-block-only eigenvectors")
	}

	inits, mids := []int{0, 1, 2, 3}, []int{0, 1, 2}
	ems, err := Emissions(ops, res.Values, res.FullVecs, inits, mids)
	if err != nil {
		t.Fatal(err)
	}
	dense := [3]backend.Mat{ops[0].BuildMatrix(), ops[1].BuildMatrix(), ops[2].BuildMatrix()}
	n := sp.Size()
	for _, e := range ems {
		if e.Init == e.Mid {
			t.Fatal("Emissions emitted a diagonal pair")
		}
		if want := res.Values[e.Init] - res.Values[e.Mid]; e.Omega != want {
			t.Errorf("omega %g, want %g", e.Omega, want)
		}
		for c := range 3 {
			var acc float64
			for i := range n {
				for j := range n {
					acc += res.FullVecs.At(i, e.Init) * dense[c].At(i, j) * res.FullVecs.At(j, e.Mid)
				}
			}
			if d := math.Abs(e.Mu[c] + acc); d > 1e-9 {
				t.Errorf("state %d→%d component %d: mu = %g, dense contraction gives %g",
					e.Init, e.Mid, c, e.Mu[c], -acc)
			}
		}
		f := 2.0 / 3.0 * e.Omega * (e.Mu[0]*e.Mu[0] + e.Mu[1]*e.Mu[1] + e.Mu[2]*e.Mu[2])
		if math.Abs(e.Osc-f) > 1e-14*math.Abs(f)+1e-18 {
			t.Errorf("oscillator strength %g, want %g", e.Osc, f)
		}
		a := 4.0 / 3.0 * math.Pow(e.Omega, 3) * (e.Mu[0]*e.Mu[0] + e.Mu[1]*e.Mu[1] + e.Mu[2]*e.Mu[2]) /
			math.Pow(SpeedOfLight, 3)
		if math.Abs(e.Rate-a) > 1e-12*math.Abs(a)+1e-24 {
			t.Errorf("Einstein A %g, want (4/3)ω³|μ|²/c³ = %g", e.Rate, a)
		}
	}
}

// TestKoopmansStateDipole ties D back to the one number in the sidecar that pyscf
// computed for itself. Take the bare 1h configuration |i⟩ as the state: its electronic
// dipole is D0 − d_ii, so the cation's dipole is μ_nuc − D0 + d_ii, which is exactly the
// neutral SCF dipole plus d_ii — the Koopmans picture of removing one electron from
// orbital i. mo.GroundStateDipole is independently gated against pyscf's scf_dip, so
// agreement here checks D0, the electron-charge sign, and the fact that sip and mo
// mean the same thing by "the dipole".
func TestKoopmansStateDipole(t *testing.T) {
	fd := h2oData(t)
	md, err := mo.ReadFile(filepath.Join("..", "..", "..", "testdata", "h2o.mo.json"))
	if err != nil {
		t.Fatalf("read sidecar: %v", err)
	}
	nocc := mp.NOcc(fd)
	sp := NewSpace(nocc, fd.NORB, nil, 0)
	ops, err := NewISRDipoles(sp, md.DipMO)
	if err != nil {
		t.Fatal(err)
	}
	neutral := md.GroundStateDipole(nocc)
	nuc := md.NuclearDipole()
	for i := range sp.BeginSat {
		orb := sp.Configs[i].Occ[0]
		x := make([]float64, sp.Size())
		x[i] = 1
		el := ElectronicDipole(ops, x)
		for c := range 3 {
			got := nuc[c] - el[c]
			want := neutral[c] + md.DipMO[c].At(orb, orb)
			if math.Abs(got-want) > 1e-10 {
				t.Errorf("hole in orbital %d, component %d: cation dipole %g, "+
					"want SCF dipole + d_ii = %g", orb, c, got, want)
			}
		}
	}
}

// TestTransitionDipoleSumRule: the eigenvectors of the dense solve are a complete
// orthonormal basis of the configuration space, so Σ_m |⟨i|D̂|m⟩|² = ‖D·X_i‖² exactly.
// This is the one check that exercises the satellite rows of *both* vectors across the
// whole spectrum at once — a truncated D, or a FullVecs whose satellite block is wrong,
// cannot satisfy it.
func TestTransitionDipoleSumRule(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping sum rule in -short mode")
	}
	sp, mx, _ := dipoleSpace(t)
	o, err := NewISRDipole(sp, randomSymmetric(sp.Norb))
	if err != nil {
		t.Fatal(err)
	}
	res := lanczos.SolveDense(mx, backend.Gonum{})
	n := sp.Size()
	col := func(k int) []float64 {
		v := make([]float64, n)
		for r := range n {
			v[r] = res.FullVecs.At(r, k)
		}
		return v
	}
	for _, i := range []int{0, 1, n / 2, n - 1} {
		xi := col(i)
		dx := make([]float64, n)
		o.Apply(dx, xi)
		var want float64
		for _, v := range dx {
			want += v * v
		}
		var got float64
		for m := range n {
			e := o.Braket(xi, col(m))
			got += e * e
		}
		if math.Abs(got-want) > 1e-8*(1+want) {
			t.Errorf("state %d: Σ_m |<i|D|m>|² = %g, ‖D·X_i‖² = %g", i, got, want)
		}
	}
}
