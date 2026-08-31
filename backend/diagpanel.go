package backend

// Accumulate a resident diagonal operator across a whole column-major panel.
//
// AxpyDiag applies a diagonal to one vector, which is what the ADC(4) satellite block
// needs: the block is applied once per Lanczos matvec, to one column at a time.
//
// MCTDH needs the same diagonal applied to many columns at once. A grid-diagonal mode
// operator — every potential, every CAP, every DVR-diagonal kinetic term — multiplies a
// sub x nspf panel of single-particle functions pointwise in the grid index, which is a
// row scaling of the panel; and the multi-dimensional potential does the same to a
// batch of primitive-grid tensors held as a gridSize x members panel. Issuing that as
// one AxpyDiag per column costs nspf (or members) dispatches for one pass over memory,
// and on a device each of those is a launch. The quantity is one call's worth of work,
// so it should be one call.
//
// AddDiagPanel is a free function over an optional capability rather than a method on
// Backend, for the same reasons GemmBatch is: no existing implementation has to change,
// and the loop over AxpyDiag stays as the correctness reference the gonum tests run.

// DiagPanelAdder is an optional capability: accumulate diag(d)*a into c for whole
// column-major panels in one call.
//
// a and c must have the same Rows and Cols, and Rows must equal d.Len(). They may have
// different leading dimensions, and c may alias a only if the two are identical views.
type DiagPanelAdder interface {
	AddDiagPanel(d Vector, a, c BlockView)
}

// AddDiagPanel computes c += diag(d) * a for column-major panels a and c — that is,
// c(i,j) += d(i) * a(i,j) — using the backend's whole-panel implementation when it has
// one and falling back to a loop of AxpyDiag over the columns when it does not.
//
// On a host backend the loop is not a slow path: it is the same passes over the same
// memory, and a gonum call costs nanoseconds. On a device it is Cols launches where one
// would do.
func AddDiagPanel(be Backend, d Vector, a, c BlockView) {
	if a.Rows != c.Rows || a.Cols != c.Cols {
		panic("backend: AddDiagPanel operand panels differ in shape")
	}
	if d.Len() != a.Rows {
		panic("backend: AddDiagPanel diagonal length does not match the panel's rows")
	}
	if dp, ok := be.(DiagPanelAdder); ok {
		dp.AddDiagPanel(d, a, c)
		return
	}
	for j := 0; j < a.Cols; j++ {
		be.AxpyDiag(d, a.Col(j), c.Col(j))
	}
}
