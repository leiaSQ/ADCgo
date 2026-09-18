package sip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// build22 makes an ADC(2,2) matrix for H2O restricted to the first norb orbitals.
// Truncating the virtual space keeps the 3h2p class small enough for the dense
// oracle tests; it is a consistent calculation in a reduced active space, not an
// approximation to the full one.
func build22(t *testing.T, norb int, v Variant) *Matrix {
	t.Helper()
	d, nocc := h2o22(t)
	eps := mp.OrbitalEnergies(d, nocc)
	sp := NewSpace22(nocc, norb, nil, 0)
	mx := New(sp, integrals.New(d, nocc, nil), eps, Order22, backend.Gonum{})
	mx.SetVariant(v)
	return mx
}

// TestADC22ReducesToADC2x is the structural gate on the whole assembly. Table I
// says ADC(2,2) differs from ADC(2)x — which is what ADCgo's -order 2 already is —
// by exactly three things: the second-order 2h1p/2h1p term, the 3h2p class, and
// (variant f only) the second-order 1h/2h1p coupling. So the 1h+2h1p corner of the
// ADC(2,2)_m matrix, with c22_2 subtracted back off, must reproduce the order-2
// matrix element for element.
//
// This checks the 1h/1h block really is order 2 and not order 3 (the c11_3 term is
// gated out by isADC22), that the coupling is c12_1 for variant m, and that the
// 2h1p enumeration of NewSpace22 agrees index-for-index with NewSpace.
func TestADC22ReducesToADC2x(t *testing.T) {
	d, nocc := h2o22(t)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)

	ref := New(NewSpace(nocc, d.NORB, nil, 0), ints, eps, 2, backend.Gonum{})
	M2 := ref.BuildMatrix()

	sp := NewSpace22(nocc, d.NORB, nil, 0)
	mx := New(sp, ints, eps, Order22, backend.Gonum{})
	mx.SetVariant(VariantM)

	if sp.BeginSat != ref.sp.BeginSat || sp.Begin3h2p != len(ref.sp.Configs) {
		t.Fatalf("1h+2h1p enumeration differs: ADC(2,2) has main=%d sat-end=%d, ADC(2)x has main=%d n=%d",
			sp.BeginSat, sp.Begin3h2p, ref.sp.BeginSat, len(ref.sp.Configs))
	}

	main, n := sp.BeginSat, sp.Begin3h2p
	var maxDiff, maxAbs float64
	for r := range n {
		for c := range n {
			got := mx.el.c11(sp.Configs[r].Occ[0], sp.Configs[c].Occ[0])
			switch {
			case r < main && c < main:
				// handled above
			case r < main:
				got = mx.el.c12_22(sp.Configs[r].Occ[0], sp.Configs[c])
			case c < main:
				got = mx.el.c12_22(sp.Configs[c].Occ[0], sp.Configs[r])
			default:
				row, col := sp.Configs[r], sp.Configs[c]
				if r == c {
					got = mx.el.c22diag(row)
				} else {
					got = mx.el.c22off(row, col)
				}
			}
			want := M2.At(r, c)
			maxAbs = math.Max(maxAbs, math.Abs(want))
			maxDiff = math.Max(maxDiff, math.Abs(got-want))
		}
	}
	t.Logf("ADC(2,2)_m 1h+2h1p corner vs ADC(2)x over %dx%d: max |element| = %.6g, max deviation = %.3g",
		n, n, maxAbs, maxDiff)
	if maxDiff > 1e-12 {
		t.Errorf("ADC(2,2) does not reduce to ADC(2)x: deviation %.3g (limit 1e-12)", maxDiff)
	}
}

// TestADC22Symmetric requires the assembled matrix to be symmetric, for all three
// variants.
func TestADC22Symmetric(t *testing.T) {
	for _, v := range []Variant{VariantM, VariantX, VariantF} {
		mx := build22(t, 8, v)
		M := mx.BuildMatrix()
		var maxAsym float64
		for i := range M.Rows {
			for j := range i {
				maxAsym = math.Max(maxAsym, math.Abs(M.At(i, j)-M.At(j, i)))
			}
		}
		t.Logf("variant %s: n=%d (1h=%d 2h1p=%d 3h2p=%d), max asymmetry %.3g",
			v, mx.sp.Size(), mx.sp.BeginSat, mx.sp.Begin3h2p-mx.sp.BeginSat,
			len(mx.sp.Sat3), maxAsym)
		if maxAsym > 1e-10 {
			t.Errorf("variant %s: asymmetry %.3g exceeds 1e-10", v, maxAsym)
		}
	}
}

// TestADC22ApplyMatchesBuild requires the block-structured operator apply to agree
// with the densely materialized matrix — the standard operator gate of this
// package, here covering the new 2h1p/3h2p coupling and the 3h2p diagonal/block.
func TestADC22ApplyMatchesBuild(t *testing.T) {
	for _, v := range []Variant{VariantM, VariantX, VariantF} {
		mx := build22(t, 8, v)
		M := mx.BuildMatrix()
		n := mx.Size()
		be := mx.be

		in := make([]float64, n)
		for i := range in {
			in[i] = math.Sin(float64(i)*0.7) + 0.3
		}
		want := make([]float64, n)
		for r := range n {
			var s float64
			for c := range n {
				s += M.At(r, c) * in[c]
			}
			want[r] = s
		}

		xv, yv := be.Upload(in), be.Upload(make([]float64, n))
		mx.ApplyFull(yv, xv)
		got := be.Download(yv)

		var maxDiff, maxAbs float64
		for i := range got {
			maxAbs = math.Max(maxAbs, math.Abs(want[i]))
			maxDiff = math.Max(maxDiff, math.Abs(got[i]-want[i]))
		}
		t.Logf("variant %s: ApplyFull vs BuildMatrix, max |component| = %.6g, max deviation = %.3g",
			v, maxAbs, maxDiff)
		if maxDiff > 1e-10 {
			t.Errorf("variant %s: ApplyFull deviates from BuildMatrix by %.3g", v, maxDiff)
		}
		mx.Release()
	}
}

// TestADC22DiagonalMatchesBuild checks the Davidson preconditioner's diagonal
// against the assembled matrix.
func TestADC22DiagonalMatchesBuild(t *testing.T) {
	for _, v := range []Variant{VariantM, VariantF} {
		mx := build22(t, 8, v)
		M := mx.BuildMatrix()
		d := mx.be.Download(mx.Diagonal(mx.be))
		var maxDiff float64
		for i := range d {
			maxDiff = math.Max(maxDiff, math.Abs(d[i]-M.At(i, i)))
		}
		t.Logf("variant %s: diagonal vs BuildMatrix, max deviation %.3g", v, maxDiff)
		if maxDiff > 1e-10 {
			t.Errorf("variant %s: diagonal deviates by %.3g", v, maxDiff)
		}
	}
}
