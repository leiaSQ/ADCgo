package backend

// Batched general GEMM over mutable panels.
//
// GemmMatBatched already batches c := alpha*op(a)*b + beta*c, but its `a` is a DeviceMat —
// an immutable block uploaded once and reused on every apply — and it has no transB. Both
// restrictions are right for the ADC secular matrices, whose operator blocks are assembled
// once per sector and applied thousands of times.
//
// They are wrong for MCTDH. There the batched call is a mode matrix applied to one index of a
// coefficient tensor (Fortran mtxxzz, a(l,j)*b(i,j,k) = c(i,l,k)), which is c_k := b_k*aᵀ —
// the matrix on the *right*, transposed — and the mode matrices are rebuilt every timestep, so
// they are panels, not immutable blocks. That is the shape GemmBatch exists for.
//
// GemmBatch is deliberately a free function over an optional capability rather than a method on
// Backend: no existing implementation has to change, and the host path (a loop over Gemm) is the
// correctness reference the gonum tests run, exactly as with BufferedDownloader and friends.

// BatchedGemm is an optional capability: apply Gemm to a whole batch in one call.
//
// Every a[i] must share Rows/Cols/Ld, likewise every b[i] and every c[i] — and the c[i] must be
// pairwise non-overlapping, because a batched GEMM runs its members concurrently and they
// accumulate. This is the same contract GemmMatBatched documents.
type BatchedGemm interface {
	GemmBatched(transA, transB bool, alpha float64, a, b []BlockView, beta float64, c []BlockView)
}

// GemmBatch computes c[i] := alpha*op(a[i])*op(b[i]) + beta*c[i] for every i, using the backend's
// batched implementation when it has one and falling back to a loop over Gemm when it does not.
//
// On a host backend the loop *is* the fast path: a gonum BLAS call costs nanoseconds, so there is
// nothing to amortize. On a device it is the slow path — one launch per member — which is why
// BatchedGemm exists.
func GemmBatch(be Backend, transA, transB bool, alpha float64, a, b []BlockView, beta float64, c []BlockView) {
	if len(a) != len(b) || len(a) != len(c) {
		panic("backend: GemmBatch operand slices differ in length")
	}
	if bg, ok := be.(BatchedGemm); ok {
		bg.GemmBatched(transA, transB, alpha, a, b, beta, c)
		return
	}
	for i := range a {
		be.Gemm(transA, transB, alpha, a[i], b[i], beta, c[i])
	}
}
