//go:build openblas

// This file makes the Gonum backend multicore: with the `openblas` build tag,
// gonum's BLAS and LAPACK engines are swapped for OpenBLAS via gonum/netlib, so
// every blas64/lapack64 call the solver already issues (GEMV in the mat-vec,
// AXPY/DOT in Lanczos, DSYEV in SymEig) runs on threaded OpenBLAS. The Gonum type
// and all solver code are unchanged — only the engine underneath differs — so the
// pure-Go reference and the OpenBLAS build share one code path.
//
// Link flags are declared here (not in netlib, which leaves library selection to
// the build) and propagate to the final link. Uses the LP64 (32-bit int)
// libraries that gonum/netlib expects, not the ILP64 (`*64_`) variants.
package backend

// #cgo LDFLAGS: -lopenblas -llapacke
import "C"

import (
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack/lapack64"
	blasnetlib "gonum.org/v1/netlib/blas/netlib"
	lapacknetlib "gonum.org/v1/netlib/lapack/netlib"
)

func init() {
	blas64.Use(blasnetlib.Implementation{})
	lapack64.Use(lapacknetlib.Implementation{})
}
