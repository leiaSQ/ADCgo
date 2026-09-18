package sip

// elements22.go — the ISR-ADC(2,2) secular-matrix blocks that non-Dyson ADC(3)
// does not already provide (Kolorenč/Averbukh, J. Chem. Phys. 152, 214107 (2020),
// Table I and Appendix A2-A22).
//
// Reused unchanged from elements.go, per the appendix's own statement that every
// block except the second-order 2h1p/2h1p one is the published non-Dyson ADC(3):
//
//	1h/1h     0,2   c11 at order 2 (k1 + c11_2; c11_3 is third order, excluded)
//	1h/2h1p   1     c12_1
//	1h/2h1p   2     c12_2   (ADC(2,2)f only)
//	2h1p/2h1p 0,1   c22diag / c22off
//
// New here:
//
//	2h1p/2h1p 2     c22_2       A9-A14
//	2h1p/3h2p 1     c23_1       A15-A17
//	3h2p/3h2p 0     c33_0       A18
//	3h2p/3h2p 1     c33_1       A19-A22   (ADC(2,2)x and f only)
//
// The zeroth- and first-order blocks are evaluated through Slater-Condon rather
// than transcribed: through first order the intermediate states ARE the HF
// configurations, so those elements are plain <D|H - E_0|D'>. See slater.go for
// why that matters — A15-A22 are dense in permutation operators, and unlike every
// other block in this package there is no reference implementation in ../ADC to
// check a transcription against.

// Variant selects one of the three ADC(2,2) schemes of Table I. They differ only
// in which of two blocks are present, so the same element code serves all three.
type Variant int

const (
	// VariantM is ADC(2,2)_m, the minimal scheme: 3h2p/3h2p at zeroth order only.
	VariantM Variant = iota
	// VariantX is ADC(2,2)_x: adds the first-order 3h2p/3h2p block.
	VariantX
	// VariantF is ADC(2,2)_f, the full scheme: adds the second-order 1h/2h1p
	// coupling on top of VariantX. The paper's own recommendation, and the only
	// variant that gets double Auger decay quantitatively right.
	VariantF
)

// String names the variant as the -adc22 flag spells it.
func (v Variant) String() string {
	switch v {
	case VariantM:
		return "m"
	case VariantX:
		return "x"
	default:
		return "f"
	}
}

// ParseVariant maps the -adc22 flag value onto a Variant.
func ParseVariant(s string) (Variant, bool) {
	switch s {
	case "m":
		return VariantM, true
	case "x":
		return VariantX, true
	case "f":
		return VariantF, true
	}
	return VariantF, false
}

// c12_22 is the 1h/2h1p coupling for ADC(2,2): first order always, plus the
// second-order term for the full variant (Table I, row "1h/2h1p"). It cannot go
// through elements.c12, whose second-order term is gated on order >= 3.
func (e *elements) c12_22(j int, cfg Config) float64 {
	v := e.c12_1(j, cfg)
	if e.variant == VariantF {
		v += e.c12_2(j, cfg)
	}
	return v
}

// c23_1 is the first-order 2h1p/3h2p coupling (A15-A17), the block that lets a
// 2h1p state decay into a 3h2p one and hence the block that makes second-order
// decay processes — double Auger, double ICD — describable at all.
func (e *elements) c23_1(row Config, col Config3) float64 {
	var sum float64
	for _, r := range e.expandDet2(row) {
		for _, c := range e.expandDet3(col) {
			sum += r.C * c.C * e.hamNO(r.E, c.E)
		}
	}
	return sum
}

// c33_0 is the zeroth-order 3h2p diagonal (A18),
//
//	eps_a + eps_b - eps_k - eps_l - eps_m,
//
// the only 3h2p/3h2p contribution in ADC(2,2)_m. It is diagonal in the
// configuration index, so the assembled operator carries it as a diagPart rather
// than a dense block — the difference between megabytes and terabytes.
func (e *elements) c33_0(cfg Config3) float64 {
	return e.eps[e.nocc+cfg.I] + e.eps[e.nocc+cfg.J] -
		e.eps[cfg.Core] - e.eps[cfg.L] - e.eps[cfg.M]
}

// c33_01 is the zeroth- plus first-order 3h2p/3h2p element (A18 plus A19-A22),
// used by ADC(2,2)_x and _f. The paper notes this block is what sets the method's
// n_occ^3 n_virt^4 cost, in matrix construction and in every matrix-vector product
// alike.
func (e *elements) c33_01(row, col Config3) float64 {
	var sum float64
	for _, r := range e.expandDet3(row) {
		for _, c := range e.expandDet3(col) {
			sum += r.C * c.C * e.hamNO(r.E, c.E)
		}
	}
	return sum
}

// isADC22 reports whether these elements belong to an ADC(2,2) matrix. It exists
// because the order-2/3 element gates read "order >= 3", and the ADC(2,2) order
// label (22) satisfies that numerically while the scheme is emphatically NOT
// third order: Table I gives its 1h/1h block as 0,2 — the third-order c11_3 term
// must stay out — and its 1h/2h1p block is variant dependent, which c12_22
// handles.
func (e *elements) isADC22() bool { return e.order == Order22 }

// Order22 is the ADC order label for ADC(2,2), as New and the -order flag take it. It
// is deliberately not 2, 3 or 4, so a space or element set can never be mistaken for
// one of the standard schemes.
const Order22 = 22
