//go:build openblas

// Divide-and-conquer symmetric eigensolver for the `openblas` build.
//
// gonum's mat.EigenSym goes through lapack64.Syev — LAPACK's dsyev, which uses QR
// iteration and is the slowest symmetric driver. gonum's lapack64 exposes no Syevd,
// and gonum/netlib does not wrap one, but OpenBLAS's LAPACKE does: dsyevd
// (divide-and-conquer) is ~11x faster at the sizes the Lanczos projected matrix
// reaches (dim ~11600 for formic acid at -blocks 200), where this phase is O(dim^3)
// and dominates everything else.
//
// Measured on a Ryzen 5600X, 6 threads, n=4000:  dsyev 39.3 s (2.2 GFLOP/s)
//
//	dsyevd 3.6 s (23.8 GFLOP/s)
//
// The prototype is declared here rather than via `#include <lapacke.h>` on purpose:
// pulling in the conda lapacke.h puts it ahead of the header gonum/netlib bundles
// for its own cgo, and the two disagree on several routine signatures (netlib's
// LAPACKE_dsyswapr_work takes 6 args, conda's takes 7). Declaring the one symbol we
// call keeps the two independent. LP64 build => lapack_int is a 32-bit C int.
package backend

/*
#cgo LDFLAGS: -llapacke -lopenblas

// LAPACKE_dsyevd: eigenvalues (ascending) and, with jobz='V', orthonormal
// eigenvectors of a real symmetric matrix. Overwrites a with the eigenvectors.
extern int LAPACKE_dsyevd(int matrix_layout, char jobz, char uplo,
                          int n, double *a, int lda, double *w);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// lapackRowMajor is LAPACKE's LAPACK_ROW_MAJOR.
const lapackRowMajor = 101

func init() { symEig = symEigLapacke }

// symEigLapacke returns the ascending eigenvalues and the eigenvectors (as columns
// of the returned Mat) of the symmetric matrix a, via LAPACKE_dsyevd.
//
// Mat is row-major, so we pass LAPACK_ROW_MAJOR with uplo='U' — matching
// symEigGonum, whose j>=i loop reads the upper triangle. On exit LAPACKE has
// transposed the result back to row-major, so a[i*n+j] is component i of
// eigenvector j: exactly the column-eigenvector convention SymEig promises.
func symEigLapacke(a Mat) ([]float64, Mat) {
	n := a.Rows
	if n == 0 {
		return nil, NewMat(0, 0)
	}
	// dsyevd overwrites its input; a.Data belongs to the caller (the projected
	// matrix T), so work on a copy.
	out := NewMat(n, n)
	copy(out.Data, a.Data)
	evals := make([]float64, n)

	info := C.LAPACKE_dsyevd(C.int(lapackRowMajor), C.char('V'), C.char('U'),
		C.int(n), (*C.double)(unsafe.Pointer(&out.Data[0])), C.int(n),
		(*C.double)(unsafe.Pointer(&evals[0])))

	switch {
	case info < 0:
		panic(fmt.Sprintf("backend: LAPACKE_dsyevd: illegal argument %d", -info))
	case info > 0:
		panic("backend: symmetric eigendecomposition failed to converge")
	}
	return evals, out
}
