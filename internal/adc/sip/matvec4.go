package sip

import "github.com/leiaSQ/ADCgo/internal/adc/backend"

// matvec4.go — assembly of the CVS IP-ADC(4) secular matrix (1h | 2h1p | 3h2p),
// dispatched from Matrix.assemble()/BuildMatrix() when order == 4. Blocks:
//
//	[1h , 1h ]  −ε_P δ − Σ_PQ (external static self-energy, SetStaticSelfEnergy) — A1 tape
//	[1h , 2h1p] coupling −(kopp1+kopp2+kopp3)          — A1 tape bit-exact
//	[1h , 3h2p] coupling −kopp4                         — A1 tape bit-exact
//	[2h1p,2h1p] c22elem4 (WERT1 3rd+4th, SUM1/3/4)      — B2 tape bit-exact
//	[2h1p,3h2p] coupling wert2elem4 (WERT2)             — B2 tape bit-exact (mod pam)
//	[3h2p,3h2p] EIGAB effective diagonal (WERT3)        — FT19 tape bit-exact (mod pam)
//
// Every block is now validated bit-exact against theADCcode on matched integrals (A1+B2
// tapes; TestADC4MatchedGate*, TestADC4EigabGate, TestADC4StaticSigmaGate). The 3h2p
// effective diagonal and the static Σ used to be unreachable — theADCcode's RSCRT1
// truncates the diagonal tape to the 1h+2h1p entries, and Σ is an *input* to adc_() that
// no tape records — so ../ADC now dumps both (ab5.F → FT19F001.ADC, egf.F →
// SIGMA_STATIC.dat). The Σ *value* remains the self-energy module's job: ADCgo consumes
// one via SetStaticSelfEnergy, it does not compute Σ(∞) itself.
//
// The reference stores the couplings and 1h/1h off-diagonal as −(value) (egf.F,
// ab3.F line 275) so the 2h1p/3h2p block diagonals stay positive (ionization
// energies); we follow that convention here. Off-diagonal blocks are placed once
// and realized both ways by the operator apply (GemvT), so the assembled M is
// symmetric. The expensive integral-sum blocks are filled row-parallel (parRows),
// which runs in this BLAS-free phase without oversubscribing the solver.
//
// isADC4 reports whether this Matrix is an order-4 CVS matrix (space built by
// NewSpace4). Guards against feeding assemble4 an order-2/3 space (Begin3h2p==0).
func (mx *Matrix) isADC4() bool { return mx.el.order == 4 && mx.sp.adc4 }

// mainBlock4 is the 1h/1h block: −ε_P δ − Σ_PQ, where Σ is the external static
// self-energy supplied via SetStaticSelfEnergy (nil → bare −ε_P). theADCcode assembles
// the same −ε−Σ (egf.F: WORK=−SIGMA−EPSI on the diagonal, AMATRX=−SIGMA off-diagonal),
// reading Σ from a separate self-energy module.
func (mx *Matrix) mainBlock4() backend.Mat {
	sp := mx.sp
	n := sp.BeginSat
	M := backend.NewMat(n, n)
	for r := range n {
		p := sp.Configs[r].Occ[0] // core hole
		M.Set(r, r, -mx.el.eps[p])
	}
	if mx.sigma != nil {
		for r := range n {
			p := sp.Configs[r].Occ[0]
			for c := range n {
				q := sp.Configs[c].Occ[0]
				M.Set(r, c, M.At(r, c)-mx.sigma(p, q))
			}
		}
	}
	return M
}

// coupling2_4 is the 1h × 2h1p coupling −(kopp1+kopp2).
func (mx *Matrix) coupling2_4() backend.Mat {
	sp := mx.sp
	ncol := sp.Begin3h2p - sp.BeginSat
	C := backend.NewMat(sp.BeginSat, ncol)
	parRows(sp.BeginSat, func(r int) {
		p := sp.Configs[r].Occ[0]
		for c := range ncol {
			cfg := sp.Configs[sp.BeginSat+c]
			C.Set(r, c, -(mx.el.kopp1(p, cfg) + mx.el.kopp2(p, cfg) + mx.el.kopp3(p, cfg)))
		}
	})
	return C
}

// coupling3_4 is the 1h × 3h2p coupling −kopp4.
func (mx *Matrix) coupling3_4() backend.Mat {
	sp := mx.sp
	ncol := len(sp.Sat3)
	C := backend.NewMat(sp.BeginSat, ncol)
	parRows(sp.BeginSat, func(r int) {
		p := sp.Configs[r].Occ[0]
		for c := range ncol {
			C.Set(r, c, -mx.el.kopp4(p, sp.Sat3[c]))
		}
	})
	return C
}

// satBlock2_4 is the symmetric 2h1p/2h1p block (c22elem4). Each worker fills a whole
// row (no cross-row writes → race-free); Hermiticity of c22elem4 makes it symmetric.
func (mx *Matrix) satBlock2_4() backend.Mat {
	sp := mx.sp
	cfgs := sp.Configs[sp.BeginSat:sp.Begin3h2p]
	n := len(cfgs)
	S := backend.NewMat(n, n)
	parRows(n, func(r int) {
		for c := range n {
			S.Set(r, c, mx.el.c22elem4(cfgs[r], cfgs[c]))
		}
	})
	return S
}

// coupling24_4 is the 2h1p × 3h2p coupling (wert2elem4 / AB5's WERT2). Rows are the
// 2h1p configs, columns the 3h2p configs; each element is the effective 1st-order
// interaction. Row-parallel (no cross-row writes).
func (mx *Matrix) coupling24_4() backend.Mat {
	sp := mx.sp
	rows := sp.Configs[sp.BeginSat:sp.Begin3h2p]
	nr, nc := len(rows), len(sp.Sat3)
	C := backend.NewMat(nr, nc)
	parRows(nr, func(r int) {
		for c := range nc {
			C.Set(r, c, mx.el.wert2elem4(rows[r], sp.Sat3[c]))
		}
	})
	return C
}

// sat3Diag is the 3h2p/3h2p block, the reference's EIGAB effective diagonal. By default
// it is the 0th-order orbital-energy sum ε_I+ε_J−ε_K−ε_L−ε_M. With WERT3 enabled
// (SetWert3, -wert3) it is the full EIGAB: that sum plus the 5th-order 3h2p-CI diagonal
// correction (elements4.go wert3elem). theADCcode evaluates WERT3 only on the diagonal
// (selec.F:120), so the block stays strictly diagonal either way — returned as its diagonal
// vector and applied via backend.AxpyDiag, never a dense n×n upload (which would be ~2.5 TB
// for pyridine's N K-edge, vs ~5 MB as a vector). ELIM's center-of-gravity fold is a
// truncation device (inactive below MAXSTA) and is not needed for the untruncated space.
//
// WERT3 is gated bit-exact against theADCcode's own EIGAB (TestADC4EigabGate, both
// sectors, multiset maxdiff ~4e-15) now that ../ADC dumps it to FT19F001.ADC. It stays
// opt-in only so that -order 4 keeps its established default behaviour; enable it with
// SetWert3/-wert3 for the order-consistent 3h2p diagonal theADCcode itself uses.
func (mx *Matrix) sat3Diag() []float64 {
	sp := mx.sp
	ep := mx.el.eps
	nocc := mx.el.nocc
	d := make([]float64, len(sp.Sat3))
	for r := range d {
		c := sp.Sat3[r]
		if mx.wert3 {
			d[r] = mx.el.wert3elem(c, c)[c.Spin-1][c.Spin-1]
		} else {
			d[r] = ep[nocc+c.I] + ep[nocc+c.J] - ep[c.Core] - ep[c.L] - ep[c.M]
		}
	}
	return d
}

// finalizeOp plans the batches and sizes the scratch slices for parts (shared with
// the order-2/3 assemble()); diags carries any purely diagonal blocks applied
// elementwise outside the GEMM batches.
func finalizeOp(parts []placement, diags []diagPart, mf []matFreePart) *assembledOp {
	batches := backend.PlanBatches(parts)
	widest := 0
	for _, b := range batches {
		if len(b.Blocks) > widest {
			widest = len(b.Blocks)
		}
	}
	return &assembledOp{
		parts: parts, batches: batches, diags: diags, mf: mf,
		sa: make([]backend.DeviceMat, widest),
		sb: make([]backend.BlockView, widest),
		sc: make([]backend.BlockView, widest),
	}
}

// assemble4 uploads the order-4 blocks as a block-sparse operator.
func (mx *Matrix) assemble4() *assembledOp {
	sp := mx.sp
	main := sp.BeginSat
	n2 := sp.Begin3h2p - main // 2h1p count
	n3 := len(sp.Sat3)
	var parts []placement
	var diags []diagPart
	var mfree []matFreePart
	add := func(m backend.Mat, r0, c0 int, diag bool) {
		parts = append(parts, placement{A: mx.be.UploadMat(m), RowOff: r0, ColOff: c0, Diag: diag})
	}
	if main > 0 {
		add(mx.mainBlock4(), 0, 0, true)
		if n2 > 0 {
			add(mx.coupling2_4(), 0, main, false)
		}
		if n3 > 0 {
			add(mx.coupling3_4(), 0, sp.Begin3h2p, false)
		}
	}
	if n2 > 0 {
		// The 2h1p×2h1p satellite block: dense by default; matrix-free only when its
		// n2²·8 residency exceeds the budget (it is far smaller than the coupling block).
		if mx.matFreeC22(int64(n2) * int64(n2) * 8) {
			mfree = append(mfree, mx.newC22MatFree())
		} else {
			add(mx.satBlock2_4(), main, main, true)
		}
	}
	if n3 > 0 {
		// The 3h2p/3h2p block is diagonal (WERT3 deferred): keep it as a resident vector
		// instead of a dense n3×n3 upload, which would be terabytes for a large sector.
		diags = append(diags, diagPart{off: sp.Begin3h2p, d: mx.be.Upload(mx.sat3Diag())})
		if n2 > 0 {
			// The 2h1p×3h2p WERT2 coupling is the memory ceiling (n2·n3·8 bytes). Apply
			// it matrix-free when requested/oversized, else assemble it densely.
			if mx.matFreeWert2(int64(n2) * int64(n3) * 8) {
				mfree = append(mfree, mx.newWert2MatFree())
			} else {
				add(mx.coupling24_4(), main, sp.Begin3h2p, false)
			}
		}
	}
	return finalizeOp(parts, diags, mfree)
}

// buildMatrix4 materializes the full symmetric order-4 matrix (both triangles).
func (mx *Matrix) buildMatrix4() backend.Mat {
	sp := mx.sp
	main := sp.BeginSat
	M := backend.NewMat(sp.Size(), sp.Size())

	mb := mx.mainBlock4()
	for r := range main {
		for c := range main {
			M.Set(r, c, mb.At(r, c))
		}
	}
	place := func(blk backend.Mat, r0, c0 int) { // off-diagonal, mirrored
		for r := range blk.Rows {
			for c := range blk.Cols {
				v := blk.At(r, c)
				M.Set(r0+r, c0+c, v)
				M.Set(c0+c, r0+r, v)
			}
		}
	}
	diag := func(blk backend.Mat, o int) {
		for r := range blk.Rows {
			for c := range blk.Cols {
				M.Set(o+r, o+c, blk.At(r, c))
			}
		}
	}
	if n2 := sp.Begin3h2p - main; n2 > 0 {
		place(mx.coupling2_4(), 0, main)
		diag(mx.satBlock2_4(), main)
	}
	if len(sp.Sat3) > 0 {
		place(mx.coupling3_4(), 0, sp.Begin3h2p)
		for r, v := range mx.sat3Diag() { // 3h2p/3h2p is diagonal
			M.Set(sp.Begin3h2p+r, sp.Begin3h2p+r, v)
		}
		if sp.Begin3h2p-main > 0 {
			place(mx.coupling24_4(), main, sp.Begin3h2p)
		}
	}
	return M
}
