package lanczos

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"

	"github.com/leiaSQ/ADCgo/backend"
)

// buildBanded returns a random real-symmetric matrix of dimension dim with half-bandwidth
// band (entries beyond the band are zero), both as a dense backend.Mat (for the dense
// oracle) and as bandStorage (for bandSymDiagFast). The bandStorage entry (col=j, i) is the
// coupling between indices j and j+i, i.e. the lower-triangle element M[j+i][j].
func buildBanded(dim, band int, seed int64) (backend.Mat, bandStorage) {
	rng := rand.New(rand.NewSource(seed))
	M := backend.NewMat(dim, dim)
	bs := newBandStorage(dim, band)
	for j := 0; j < dim; j++ {
		for i := 0; i <= band && j+i < dim; i++ {
			v := rng.NormFloat64()
			bs.set(j, i, v)
			M.Set(j+i, j, v)
			M.Set(j, j+i, v)
		}
	}
	return M, bs
}

// TestBandSymDiagFastEigenvalues checks the ported bnd2td/tddiag eigenvalues against the
// dense SymEig oracle across several bandwidths, including the degenerate band==0 and
// band==1 control-flow branches.
func TestBandSymDiagFastEigenvalues(t *testing.T) {
	be := backend.Gonum{}
	for _, tc := range []struct{ dim, band int }{
		{1, 0}, {8, 0}, {8, 1}, {12, 2}, {30, 5}, {40, 7}, {50, 12},
	} {
		M, bs := buildBanded(tc.dim, tc.band, int64(1000+tc.dim*100+tc.band))
		want, _ := be.SymEig(M) // ascending
		got, _ := bandSymDiagFast(bs)
		if len(got) != tc.dim {
			t.Fatalf("dim=%d band=%d: got %d evals, want %d", tc.dim, tc.band, len(got), tc.dim)
		}
		for k := range want {
			if math.Abs(got[k]-want[k]) > 1e-9 {
				t.Errorf("dim=%d band=%d: eval[%d]=%.12f, want %.12f (Δ%.2e)",
					tc.dim, tc.band, k, got[k], want[k], got[k]-want[k])
			}
		}
	}
}

// TestBandSymDiagFastPartialVectors checks that the 2*band-row partial eigenvectors match
// the top and bottom band rows of the dense eigenvectors (up to a per-eigenvector sign),
// which is the property Mode B relies on to read pole strengths without the basis.
func TestBandSymDiagFastPartialVectors(t *testing.T) {
	be := backend.Gonum{}
	const dim, band = 36, 6
	M, bs := buildBanded(dim, band, 424242)
	evalsD, evecsD := be.SymEig(M)
	evals, z := bandSymDiagFast(bs)
	nm := 2 * band

	for k := 0; k < dim; k++ {
		// Skip near-degenerate eigenvalues: their eigenvectors are only defined up to a
		// rotation within the degenerate subspace, so a row-by-row match is not meaningful.
		degenerate := false
		for j := 0; j < dim; j++ {
			if j != k && math.Abs(evalsD[j]-evalsD[k]) < 1e-6 {
				degenerate = true
				break
			}
		}
		if degenerate {
			continue
		}
		// Reference top/bottom slices from the dense eigenvector (column k).
		ref := make([]float64, nm)
		for r := 0; r < band; r++ {
			ref[r] = evecsD.At(r, k)
			ref[band+r] = evecsD.At(dim-band+r, k)
		}
		got := z[k*nm : (k+1)*nm]
		// Fix the global sign by the largest-magnitude reference component.
		pivot := 0
		for r := 1; r < nm; r++ {
			if math.Abs(ref[r]) > math.Abs(ref[pivot]) {
				pivot = r
			}
		}
		sign := 1.0
		if ref[pivot]*got[pivot] < 0 {
			sign = -1.0
		}
		for r := 0; r < nm; r++ {
			if math.Abs(sign*got[r]-ref[r]) > 1e-7 {
				t.Errorf("eval %d (%.6f): partial vec row %d = %.9f, want %.9f",
					k, evals[k], r, sign*got[r], ref[r])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Frozen reference oracle
// ---------------------------------------------------------------------------
//
// bandSymDiagFastRef / bnd2tdRef / tddiagRef are VERBATIM copies of the serial
// bandSymDiagFast / bnd2td / tddiag as of commit e14bfd3 (2026-09-28), before the
// eigensolver was parallelized. Only the three function names differ; the bodies were
// extracted mechanically from bandeig.go so the copy is provably identical.
//
// They exist because the parallelization is required to be bit-for-bit output-preserving
// and nothing else in the tree can check that: no golden file pins bandeig output, the
// dense oracle below compares to 1e-9, and lowmem_checkpoint_test.go's resume check
// compares two runs of the SAME binary. TestBandSymDiagFastBitExactVsReference is the
// only gate that would notice a changed rounding, an operand reorder or an FMA fusion.
//
// Do NOT "fix" or modernize these copies, and do not re-derive them from the production
// functions: their whole value is that they are the pre-change arithmetic. If a genuine
// numerical bug is ever found in the reference algorithm, fix bandeig.go, re-extract these
// from the commit that introduced the fix, and say so here.

func bandSymDiagFastRef(bs bandStorage) (evals []float64, z []float64) {
	dim, band := bs.dim, bs.band
	if dim == 0 {
		return nil, nil
	}
	b1 := band + 1
	nm := 2 * band

	// Fortran a(n=dim, mb=b1), column-major: A(row,col) 1-based at af[(col-1)*dim + row-1].
	// Repack the column storage into it exactly as lanczos_util.cpp:114-116:
	//   a[i+j+dim*(b1-i-1)] = mat[i+j*b1]     (i in [0,b1), j in [0,dim-i))
	af := make([]float64, b1*dim)
	for i := range b1 {
		for j := 0; j < dim-i; j++ {
			af[i+j+dim*(b1-i-1)] = bs.at(j, i)
		}
	}

	d := make([]float64, dim)
	e := make([]float64, dim)
	e2 := make([]float64, dim)
	zf := make([]float64, nm*dim)

	bnd2tdRef(nm, dim, b1, af, d, e, e2, zf)
	tddiagRef(nm, dim, d, e, zf)

	return d, zf
}

func bnd2tdRef(nm, n, mb int, a, d, e, e2, z []float64) {
	const (
		half   = 0.5
		two    = 2.0
		dmin   = 5.421010862427522e-20  // 2^-64
		dminrt = 2.3283064365386963e-10 // 2^-32
	)
	// A(i,j) / Z(k,j), 1-based like Fortran.
	A := func(i, j int) float64 { return a[(j-1)*n+(i-1)] }
	setA := func(i, j int, v float64) { a[(j-1)*n+(i-1)] = v }
	Z := func(k, j int) float64 { return z[(j-1)*nm+(k-1)] }
	setZ := func(k, j int, v float64) { z[(j-1)*nm+(k-1)] = v }

	for j := 1; j <= n; j++ {
		d[j-1] = 1
	}
	nm2 := nm / 2
	// z already zeroed (Go make); set the two identity strips.
	for j := 1; j <= nm2; j++ {
		setZ(j, j, 1)
		setZ(j+nm2, n-nm2+j, 1)
	}

	m1 := mb - 1
	switch {
	case m1 < 1: // m1-1 < 0  → label 900: diagonal only
		for j := 1; j <= n; j++ {
			d[j-1] = A(j, mb)
			e[j-1] = 0
			e2[j-1] = 0
		}
		return
	case m1 == 1: // m1-1 == 0 → label 800 directly (already tridiagonal)
	default: // m1 > 1 → general band reduction (label 70), then fall through to 800
		n2 := n - 2
		for k := 1; k <= n2; k++ {
			maxr := min(m1, n-k)
			for r1 := 2; r1 <= maxr; r1++ {
				r := maxr + 2 - r1
				kr := k + r
				mr := mb - r
				g := A(kr, mr)
				setA(kr-1, 1, A(kr-1, mr+1))
				ugl := k
				for j := kr; j <= n; j += m1 {
					j1 := j - 1
					j2 := j1 - 1
					if g == 0 {
						break
					}
					b1v := A(j1, 1) / g
					b2 := b1v * d[j1-1] / d[j-1]
					s2 := 1.0 / (1.0 + b1v*b2)
					if s2 < half {
						b1v = g / A(j1, 1)
						b2 = b1v * d[j-1] / d[j1-1]
						c2 := 1.0 - s2
						d[j1-1] = c2 * d[j1-1]
						d[j-1] = c2 * d[j-1]
						f1 := two * A(j, m1)
						f2 := b1v * A(j1, mb)
						setA(j, m1, -b2*(b1v*A(j, m1)-A(j, mb))-f2+A(j, m1))
						setA(j1, mb, b2*(b2*A(j, mb)+f1)+A(j1, mb))
						setA(j, mb, b1v*(f2-f1)+A(j, mb))
						for l := ugl; l <= j2; l++ {
							i2 := mb - j + l
							u := A(j1, i2+1) + b2*A(j, i2)
							setA(j, i2, -b1v*A(j1, i2+1)+A(j, i2))
							setA(j1, i2+1, u)
						}
						ugl = j
						setA(j1, 1, A(j1, 1)+b2*g)
						if j != n {
							maxl := min(m1, n-j1)
							for l := 2; l <= maxl; l++ {
								i1 := j1 + l
								i2 := mb - l
								u := A(i1, i2) + b2*A(i1, i2+1)
								setA(i1, i2+1, -b1v*A(i1, i2)+A(i1, i2+1))
								setA(i1, i2, u)
							}
							i1 := j + m1
							if i1 <= n {
								g = b2 * A(i1, 1)
							}
						}
						for l := 1; l <= nm; l++ {
							u := Z(l, j1) + b2*Z(l, j)
							setZ(l, j, -b1v*Z(l, j1)+Z(l, j))
							setZ(l, j1, u)
						}
					} else {
						u := d[j1-1]
						d[j1-1] = s2 * d[j-1]
						d[j-1] = s2 * u
						f1 := two * A(j, m1)
						f2 := b1v * A(j, mb)
						uu := b1v*(f2-f1) + A(j1, mb)
						setA(j, m1, b2*(b1v*A(j, m1)-A(j1, mb))+f2-A(j, m1))
						setA(j1, mb, b2*(b2*A(j1, mb)+f1)+A(j, mb))
						setA(j, mb, uu)
						for l := ugl; l <= j2; l++ {
							i2 := mb - j + l
							u2 := b2*A(j1, i2+1) + A(j, i2)
							setA(j, i2, -A(j1, i2+1)+b1v*A(j, i2))
							setA(j1, i2+1, u2)
						}
						ugl = j
						setA(j1, 1, b2*A(j1, 1)+g)
						if j != n {
							maxl := min(m1, n-j1)
							for l := 2; l <= maxl; l++ {
								i1 := j1 + l
								i2 := mb - l
								u2 := b2*A(i1, i2) + A(i1, i2+1)
								setA(i1, i2+1, -A(i1, i2)+b1v*A(i1, i2+1))
								setA(i1, i2, u2)
							}
							i1 := j + m1
							if i1 <= n {
								g = A(i1, 1)
								setA(i1, 1, b1v*A(i1, 1))
							}
						}
						for l := 1; l <= nm; l++ {
							u2 := b2*Z(l, j1) + Z(l, j)
							setZ(l, j, -Z(l, j1)+b1v*Z(l, j))
							setZ(l, j1, u2)
						}
					}
				}
			}
			// Periodic underflow rescale every 64 columns (bnd2tdRef.f:124-145).
			if k%64 == 0 {
				for j := k; j <= n; j++ {
					if d[j-1] >= dmin {
						continue
					}
					maxl := max(1, mb+1-j)
					for l := maxl; l <= m1; l++ {
						setA(j, l, dminrt*A(j, l))
					}
					if j != n {
						maxl2 := min(m1, n-j)
						for l := 1; l <= maxl2; l++ {
							i1 := j + l
							i2 := mb - l
							setA(i1, i2, dminrt*A(i1, i2))
						}
					}
					for l := 1; l <= nm; l++ {
						setZ(l, j, dminrt*Z(l, j))
					}
					setA(j, mb, dmin*A(j, mb))
					d[j-1] = d[j-1] / dmin
				}
			}
		}
	}

	// Label 800: form the tridiagonal (d, e, e2) and scale z.
	for j := 2; j <= n; j++ {
		e[j-1] = math.Sqrt(d[j-1])
	}
	for j := 2; j <= n; j++ {
		for k := 1; k <= nm; k++ {
			setZ(k, j, e[j-1]*Z(k, j))
		}
	}
	u := 1.0
	for j := 2; j <= n; j++ {
		setA(j, m1, u*e[j-1]*A(j, m1))
		u = e[j-1]
		e2[j-1] = A(j, m1) * A(j, m1)
		setA(j, mb, d[j-1]*A(j, mb))
		d[j-1] = A(j, mb)
		e[j-1] = A(j, m1)
	}
	d[0] = A(1, mb)
	e[0] = 0
	e2[0] = 0
}

func tddiagRef(nm, n int, d, e, z []float64) int {
	const (
		machep = 2.22045e-16
		mxiter = 30
	)
	Z := func(k, j int) float64 { return z[(j-1)*nm+(k-1)] }
	setZ := func(k, j int, v float64) { z[(j-1)*nm+(k-1)] = v }

	if n == 1 {
		return 0
	}
	for i := 2; i <= n; i++ {
		e[i-2] = e[i-1]
	}
	f := 0.0
	b := 0.0
	e[n-1] = 0
	for l := 1; l <= n; l++ {
		j := 0
		h := machep * (math.Abs(d[l-1]) + math.Abs(e[l-1]))
		if b < h {
			b = h
		}
		// Look for a small sub-diagonal element (finds m; only its use as p=d(m) below matters).
		m := l
		for ; m <= n; m++ {
			if math.Abs(e[m-1]) <= b {
				break
			}
		}
		for math.Abs(e[l-1]) > b && j < mxiter {
			j++
			l1 := l + 1
			g := d[l-1]
			p := (d[l1-1] - g) / (e[l-1] * 2)
			r := math.Sqrt(p*p + 1)
			d[l-1] = e[l-1] / (p + math.Copysign(r, p))
			h = g - d[l-1]
			for i := l1; i <= n; i++ {
				d[i-1] -= h
			}
			f += h
			p = d[m-1]
			c := 1.0
			s := 0.0
			mml := m - l
			for ii := 1; ii <= mml; ii++ {
				i := m - ii
				g = c * e[i-1]
				h = c * p
				if math.Abs(p) >= math.Abs(e[i-1]) {
					c = e[i-1] / p
					r = math.Sqrt(c*c + 1)
					e[i] = s * p * r
					s = c / r
					c = 1 / r
				} else {
					c = p / e[i-1]
					r = math.Sqrt(c*c + 1)
					e[i] = s * e[i-1] * r
					s = 1 / r
					c = c * s
				}
				p = c*d[i-1] - s*g
				d[i] = h + s*(c*g+s*d[i-1])
				for k := 1; k <= nm; k++ {
					hh := Z(k, i+1)
					setZ(k, i+1, s*Z(k, i)+c*hh)
					setZ(k, i, c*Z(k, i)-s*hh)
				}
			}
			e[l-1] = s * p
			d[l-1] = c * p
		}
		if j == mxiter {
			return l
		}
		d[l-1] += f
	}

	// Ascending selection sort of eigenvalues, carrying the z columns along.
	for ii := 2; ii <= n; ii++ {
		i := ii - 1
		k := i
		p := d[i-1]
		for jj := ii; jj <= n; jj++ {
			if d[jj-1] >= p {
				continue
			}
			k = jj
			p = d[jj-1]
		}
		if k == i {
			continue
		}
		d[k-1] = d[i-1]
		d[i-1] = p
		for jj := 1; jj <= nm; jj++ {
			pp := Z(jj, i)
			setZ(jj, i, Z(jj, k))
			setZ(jj, k, pp)
		}
	}
	return 0
}

// ---------------------------------------------------------------------------
// Bit-exactness gate
// ---------------------------------------------------------------------------

// bandEigCases are the {dim,band} shapes the bit-exactness gate runs. The first seven are
// the dense-oracle cases above (they cover the band==0 / band==1 / small-band control-flow
// branches); the last three exist to reach code the small cases never touch:
//
//	{200,2} and {512,5} pass k = 64, so they exercise the periodic underflow rescale
//	  (bnd2td.f:124-145) — measured 549 rescales at {512,5}, zero at every case above,
//	  because {50,12} stops at k = 48;
//	{3712,37} has dim/band = 100, the production ratio, so its j-chases are ~100 steps
//	  long instead of the 4 steps {50,12} manages. Chase length is what the A-side
//	  dependency argument turns on, so a short-chase-only suite proves nothing about it.
//
// long marks the cases skipped under -short (a few seconds each, doubled by the oracle).
var bandEigCases = []struct {
	dim, band int
	long      bool
}{
	{1, 0, false}, {8, 0, false}, {8, 1, false}, {12, 2, false},
	{30, 5, false}, {40, 7, false}, {50, 12, false},
	{200, 2, false}, {512, 5, true}, {3712, 37, true},
}

// TestBandSymDiagFastBitExactVsReference is the primary gate on every change to bandeig.go:
// the solver must reproduce the frozen pre-parallelization arithmetic EXACTLY, not to a
// tolerance. Mode B feeds these eigenvalues and eigenvector rows straight into pole
// strengths, so a changed last bit is a changed published number; and with Lanczos ghosts
// the eigenvalue list contains near-duplicates whose ordering a rounding change can flip,
// which reorders whole states rather than perturbing them.
//
// Comparison is by != on the raw float64s. A tolerance here would defeat the point.
func TestBandSymDiagFastBitExactVsReference(t *testing.T) {
	for _, tc := range bandEigCases {
		if tc.long && testing.Short() {
			continue
		}
		_, bs := buildBanded(tc.dim, tc.band, int64(1000+tc.dim*100+tc.band))
		_, ref := buildBanded(tc.dim, tc.band, int64(1000+tc.dim*100+tc.band))
		wantD, wantZ := bandSymDiagFastRef(ref)
		gotD, gotZ := bandSymDiagFast(bs)
		if len(gotD) != len(wantD) || len(gotZ) != len(wantZ) {
			t.Fatalf("dim=%d band=%d: got %d evals / %d z, want %d / %d",
				tc.dim, tc.band, len(gotD), len(gotZ), len(wantD), len(wantZ))
		}
		bad := 0
		for k := range wantD {
			if gotD[k] != wantD[k] {
				if bad++; bad <= 5 {
					t.Errorf("dim=%d band=%d: eval[%d] = %.17g, want %.17g",
						tc.dim, tc.band, k, gotD[k], wantD[k])
				}
			}
		}
		for i := range wantZ {
			if gotZ[i] != wantZ[i] {
				if bad++; bad <= 5 {
					t.Errorf("dim=%d band=%d: z[%d] (eigenvector %d row %d) = %.17g, want %.17g",
						tc.dim, tc.band, i, i/(2*tc.band), i%(2*tc.band), gotZ[i], wantZ[i])
				}
			}
		}
		if bad > 5 {
			t.Errorf("dim=%d band=%d: %d differing values in total", tc.dim, tc.band, bad)
		}
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------
//
// NOTE — these do NOT reproduce the production memory regime, and no seconds-scale
// benchmark can. Production is dim = 308000, band = 3079: af and z are ~7.6 GB each and the
// per-k working set is ~39 MB, so every A-side access misses to DRAM. Reproducing that
// footprint means reproducing production's cost (~days). Tier 1 below is for tuning and
// regression only; treat any extrapolation from it to the cluster as an upper bound on
// speed. BenchmarkBandSymDiagFastTier2 is the honest scaling check — see its comment.
//
// dim/band = 100 is held fixed because chase length, not dim, sets the A-side behaviour.

// BenchmarkBandSymDiagFast is the tier-1 end-to-end benchmark: the smallest shape with
// production's chase length (~100 steps) that also reaches the periodic rescale (6433 times).
func BenchmarkBandSymDiagFast(b *testing.B) {
	_, proto := buildBanded(3712, 37, 7)
	for b.Loop() {
		bs := bandStorage{data: append([]float64(nil), proto.data...), dim: proto.dim, band: proto.band}
		bandSymDiagFast(bs)
	}
}

// BenchmarkBnd2tdSpine times bnd2td asking for ZERO accumulated eigenvector rows. Every z op is
// then dropped at the recorder, so what remains is the sequential spine — the band→tridiagonal
// reduction of A, which no amount of z threading can shorten. Against BenchmarkBnd2tdFull it
// measures the Amdahl ceiling of a z-only parallelization directly, with no code change.
//
// Measured 2026-09-28 on a 64-core Helix node at dim=3712 band=37: bnd2td 2.03 s full / 1.20 s
// spine, tddiag 2.28 s full / 0.27 s spine — spine share 34%, so a z-only fix caps at 2.95x
// however many cores it gets. The same probe at {9240,92} gave 3.11x and at {4620,231} 3.66x.
// That measurement is why the A-side row walks are threaded too and not just the rotations.
func BenchmarkBnd2tdSpine(b *testing.B) { benchBnd2td(b, 0) }

// BenchmarkBnd2tdFull is BenchmarkBnd2tdSpine with the real row count, for the ratio.
func BenchmarkBnd2tdFull(b *testing.B) { benchBnd2td(b, -1) }

func benchBnd2td(b *testing.B, rows int) {
	const dim, band = 3712, 37
	_, bs := buildBanded(dim, band, 7)
	if rows < 0 {
		rows = band
	}
	b1 := band + 1
	proto := make([]float64, b1*dim)
	for i := range b1 {
		for j := 0; j < dim-i; j++ {
			proto[i+j+dim*(b1-i-1)] = bs.at(j, i)
		}
	}
	af := make([]float64, len(proto))
	d := make([]float64, dim)
	e := make([]float64, dim)
	e2 := make([]float64, dim)
	pool := parallel.NewFixedPool(runtime.GOMAXPROCS(0))
	defer pool.Close()
	for b.Loop() {
		copy(af, proto)
		za := testZAccum(b, dim, bandEigOpts{topRows: rows, botRows: rows})
		bnd2td(za, dim, b1, af, d, e, e2, pool, 0, nil, nil, 0, nil, 0)
		za.flush()
	}
}

// BenchmarkTddiagZ times the QL sweep, which is ~90% z rotation at these shapes and so is
// the part Phase 2's deferred tape should very nearly eliminate. The bnd2td reduction is
// hoisted out of the timed region.
func BenchmarkTddiagZ(b *testing.B) {
	const dim, band = 3712, 37
	_, bs := buildBanded(dim, band, 7)
	b1 := band + 1
	af := make([]float64, b1*dim)
	for i := range b1 {
		for j := 0; j < dim-i; j++ {
			af[i+j+dim*(b1-i-1)] = bs.at(j, i)
		}
	}
	d0 := make([]float64, dim)
	e0 := make([]float64, dim)
	pool := parallel.NewFixedPool(runtime.GOMAXPROCS(0))
	defer pool.Close()
	za0 := testZAccum(b, dim, bandEigOpts{topRows: band, botRows: band})
	bnd2td(za0, dim, b1, af, d0, e0, make([]float64, dim), pool, 0, nil, nil, 0, nil, 0)
	za0.flush()
	d := make([]float64, dim)
	e := make([]float64, dim)
	za := testZAccum(b, dim, bandEigOpts{topRows: band, botRows: band})
	for b.Loop() {
		copy(d, d0)
		copy(e, e0)
		copy(za.z, za0.z)
		tddiag(za, dim, d, e)
	}
}

// BenchmarkBandSymDiagFastTier2 fixes band at the production 3079 and reports the frozen serial
// reference against both the symmetric and the narrow (production) form, so the speedup is a ratio
// measured in one process at the production BAND — which is what sets the accumulated row count
// (3080 narrow) and the per-column footprint (~39 MB), and so the cache behaviour that tier 1,
// whose arrays fit in L3, cannot reproduce.
//
// It is not the production DIM. Cost is linear in dim at fixed band, so two dims a factor 2 apart
// should differ by a factor 2; if the larger comes out worse than that, DRAM effects have begun and
// any extrapolation to dim = 308000 is optimistic. TIER2_DIMS selects them (default just the
// smaller, which is already ~20 minutes for the reference).
//
// Not run by default: minutes per iteration and gigabytes of af and z.
func BenchmarkBandSymDiagFastTier2(b *testing.B) {
	if testing.Short() {
		b.Skip("tier-2: minutes per iteration; run explicitly without -short")
	}
	const band = 3079
	dims := []int{9240}
	if v := os.Getenv("TIER2_DIMS"); v != "" {
		dims = nil
		for _, f := range strings.Split(v, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				b.Fatalf("TIER2_DIMS=%q: %v", v, err)
			}
			dims = append(dims, n)
		}
	}
	for _, dim := range dims {
		_, proto := buildBanded(dim, band, 7)
		fresh := func() bandStorage {
			return bandStorage{data: append([]float64(nil), proto.data...), dim: dim, band: band}
		}
		b.Run(fmt.Sprintf("dim=%d/ref", dim), func(b *testing.B) {
			for b.Loop() {
				bandSymDiagFastRef(fresh())
			}
		})
		// The symmetric form only when its two strips fit without overlapping; past 2*band > dim it
		// carries matrix rows twice, which is legal but is not a shape production ever asks for.
		if 2*band <= dim {
			b.Run(fmt.Sprintf("dim=%d/wide", dim), func(b *testing.B) {
				for b.Loop() {
					bandSymDiagFast(fresh())
				}
			})
		}
		// Two accumulated rows, i.e. essentially no eigenvector work at all: what remains is the
		// band-matrix spine, which is the part a GPU port of the rotations would leave on the CPU.
		// ref minus this is the share such a port could address.
		b.Run(fmt.Sprintf("dim=%d/spine", dim), func(b *testing.B) {
			for b.Loop() {
				bandSymDiagFastOpts(fresh(), bandEigOpts{topRows: 1, botRows: 1, workers: 1})
			}
		})
		// The production call: a `main` top strip and a last-block bottom strip, each of width b,
		// where band = 2b-1. Run at one worker as well, because the two effects have to be told
		// apart: narrowing the accumulated rows halves the work whatever the core count, while
		// threading it is the part that can be bounded by memory bandwidth rather than cores.
		// narrow-w1 against ref isolates the first; narrow against narrow-w1 isolates the second.
		main := (band + 1) / 2
		for _, w := range []int{1, 0} {
			name := fmt.Sprintf("dim=%d/narrow", dim)
			if w == 1 {
				name += "-w1"
			}
			b.Run(name, func(b *testing.B) {
				for b.Loop() {
					bandSymDiagFastOpts(fresh(), bandEigOpts{topRows: main, botRows: main, workers: w})
				}
			})
		}
	}
}

// testZAccum builds an accumulator with its own pool, for the tests and benchmarks that drive
// bnd2td/tddiag/replay directly instead of going through bandSymDiagFastOpts.
func testZAccum(t testing.TB, n int, o bandEigOpts) *zAccum {
	workers := o.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	pool := parallel.NewFixedPool(workers)
	t.Cleanup(pool.Close)
	return newZAccum(n, o, pool)
}

// bandEigCmpRef compares one bandSymDiagFastOpts result against the frozen serial reference,
// which always accumulates the symmetric 2*band rows with a tight leading dimension. got must
// have been produced with topRows = botRows = band; ld may be padded.
func bandEigCmpRef(t *testing.T, tag string, dim, band, ld int, gotD, gotZ, wantD, wantZ []float64) {
	t.Helper()
	nm := 2 * band
	bad := 0
	for k := range wantD {
		if gotD[k] != wantD[k] {
			if bad++; bad <= 5 {
				t.Errorf("%s: eval[%d] = %.17g, want %.17g", tag, k, gotD[k], wantD[k])
			}
		}
	}
	for k := range dim {
		for r := range nm {
			if g, w := gotZ[k*ld+r], wantZ[k*nm+r]; g != w {
				if bad++; bad <= 5 {
					t.Errorf("%s: z[eigenvector %d][row %d] = %.17g, want %.17g", tag, k, r, g, w)
				}
			}
		}
	}
	if bad > 5 {
		t.Errorf("%s: %d differing values in total", tag, bad)
	}
}

// TestBandEigBitExactAcrossWorkersAndBatch is the proof that deferring the eigenvector writes
// onto a tape and replaying them over row blocks is a loop interchange and not a
// reassociation: every worker count and every batch size must land on the SAME bits as the
// serial reference.
//
// The shapes are chosen for what they stress, not for size.
//
//	{260,128} has 256 accumulated rows, so zRowMin lets 8 workers run and the row blocks are
//	  genuinely split, while the matrix stays small enough to sweep every combination. It carries
//	  the batch sweep: batch = 1 is the harshest ordering test there is, since it flushes between
//	  every single op and would expose any hidden dependence on ops being replayed together.
//	{502,250} has 500 rows — up to 15 workers, and a last row block that is partial because 500
//	  is not a multiple of zRowAlign.
//	{3712,37} has dim/band = 100, the production ratio, so its chases are ~100 steps long. Chase
//	  length is what the band-matrix deferral's dependency argument turns on, so this is the case
//	  that actually tests aTape.
//
// The odd worker counts catch off-by-one row blocks.
func TestBandEigBitExactAcrossWorkersAndBatch(t *testing.T) {
	cases := []struct {
		dim, band int
		workers   []int
		batches   []int
		long      bool
	}{
		{260, 128, []int{1, 2, 3, 7, 0}, []int{1, 2, 17, 1024, 0}, false},
		{502, 250, []int{1, 3, 0}, []int{2, 0}, false},
		{3712, 37, []int{1, 2, 0}, []int{0}, true},
	}
	for _, tc := range cases {
		if tc.long && testing.Short() {
			continue
		}
		seed := int64(1000 + tc.dim*100 + tc.band)
		_, ref := buildBanded(tc.dim, tc.band, seed)
		wantD, wantZ := bandSymDiagFastRef(ref)
		for _, w := range tc.workers {
			for _, batch := range tc.batches {
				_, bs := buildBanded(tc.dim, tc.band, seed)
				gotD, gotZ, ld, _ := bandSymDiagFastOpts(bs, bandEigOpts{
					topRows: tc.band, botRows: tc.band, workers: w, batch: batch,
				})
				tag := fmt.Sprintf("dim=%d band=%d workers=%d batch=%d", tc.dim, tc.band, w, batch)
				bandEigCmpRef(t, tag, tc.dim, tc.band, ld, gotD, gotZ, wantD, wantZ)
			}
		}
	}
}

// TestBandEigDeterministicAcrossRuns pins run-to-run determinism at the default worker count.
// The replay's row partition is static (parallel.Chunks), not work-stealing, so goroutine
// scheduling cannot reach the arithmetic; this test is what would notice if that ever changed.
func TestBandEigDeterministicAcrossRuns(t *testing.T) {
	const dim, band = 260, 128
	var firstD, firstZ []float64
	for run := range 5 {
		_, bs := buildBanded(dim, band, 99)
		d, z, _, _ := bandSymDiagFastOpts(bs, bandEigOpts{topRows: band, botRows: band})
		if run == 0 {
			firstD, firstZ = d, z
			continue
		}
		for i := range firstD {
			if d[i] != firstD[i] {
				t.Fatalf("run %d: eval[%d] = %.17g, run 0 gave %.17g", run, i, d[i], firstD[i])
			}
		}
		for i := range firstZ {
			if z[i] != firstZ[i] {
				t.Fatalf("run %d: z[%d] = %.17g, run 0 gave %.17g", run, i, z[i], firstZ[i])
			}
		}
	}
}

// TestBandEigNarrowRowsMatchWideRows is the real proof that z is a write-only accumulator:
// asking for fewer eigenvector rows must not change the rows that ARE asked for. If any scalar
// in bnd2td or tddiag were ever read back out of z, dropping rows would perturb it and these
// comparisons — which are on bits, not tolerances — would fail.
//
// Row mapping: the narrow run's top row r is the wide run's row r, and its bottom row r is
// original matrix row dim-botRows+r, which the wide run holds at 2*band-botRows+r.
func TestBandEigNarrowRowsMatchWideRows(t *testing.T) {
	for _, tc := range []struct{ dim, band int }{{40, 7}, {200, 2}, {512, 5}, {260, 128}} {
		_, wide := buildBanded(tc.dim, tc.band, 31337)
		wideD, wideZ, wideLD, _ := bandSymDiagFastOpts(wide, bandEigOpts{topRows: tc.band, botRows: tc.band})
		for _, rr := range [][2]int{{1, 1}, {tc.band, 1}, {1, tc.band}, {(tc.band + 1) / 2, tc.band / 2}} {
			top, bot := rr[0], rr[1]
			if top < 1 || bot < 1 || top > tc.band || bot > tc.band || top+bot > tc.dim {
				continue
			}
			_, nb := buildBanded(tc.dim, tc.band, 31337)
			d, z, ld, _ := bandSymDiagFastOpts(nb, bandEigOpts{topRows: top, botRows: bot})
			for k := range wideD {
				if d[k] != wideD[k] {
					t.Fatalf("dim=%d band=%d top=%d bot=%d: eval[%d] = %.17g, wide gave %.17g",
						tc.dim, tc.band, top, bot, k, d[k], wideD[k])
				}
			}
			for k := range tc.dim {
				for r := range top {
					if g, w := z[k*ld+r], wideZ[k*wideLD+r]; g != w {
						t.Fatalf("dim=%d band=%d top=%d bot=%d: z[%d] top row %d = %.17g, wide gave %.17g",
							tc.dim, tc.band, top, bot, k, r, g, w)
					}
				}
				for r := range bot {
					if g, w := z[k*ld+top+r], wideZ[k*wideLD+2*tc.band-bot+r]; g != w {
						t.Fatalf("dim=%d band=%d top=%d bot=%d: z[%d] bottom row %d = %.17g, wide gave %.17g",
							tc.dim, tc.band, top, bot, k, r, g, w)
					}
				}
			}
		}
	}
}

// TestZTapeOpsMatchNaive checks each deferred op kind on its own against a literal
// transcription of the Fortran loop it stands for, over a range of row counts and worker
// counts. The end-to-end oracle above covers all four kinds together, but only at whatever
// mixture a given matrix happens to produce; this pins each kind individually, at row counts
// that are and are not multiples of zRowAlign, and at worker counts that split a column into
// several blocks.
func TestZTapeOpsMatchNaive(t *testing.T) {
	// n only has to exceed the largest row count (newZAccum rejects strips wider than the
	// dimension) and leave room for the nine columns the ops below touch.
	const n = 128
	for _, rows := range []int{1, 7, 8, 9, 32, 33, 64, 100} {
		for _, workers := range []int{1, 2, 3, 0} {
			rng := rand.New(rand.NewSource(int64(rows*100 + workers)))
			top := (rows + 1) / 2
			za := testZAccum(t, n, bandEigOpts{topRows: top, botRows: rows - top, workers: workers, batch: 3})
			// A dense random accumulator, and a naive mirror of it with a tight leading dimension.
			naive := make([]float64, rows*n)
			for j := 1; j <= n; j++ {
				col := za.col(j)
				for r := range col {
					v := rng.NormFloat64()
					col[r] = v
					naive[(j-1)*rows+r] = v
				}
			}
			nc := func(j int) []float64 { return naive[(j-1)*rows : (j-1)*rows+rows] }

			b1v, b2, c, s, f := 0.3125, -1.75, 0.6, -0.8, 0.25
			// zOpRotA — bnd2td.f s2 < 0.5 form, on columns 3,4.
			za.rotA(4, b1v, b2)
			for l, cj1, cj := 0, nc(3), nc(4); l < rows; l++ {
				u := cj1[l] + b2*cj[l]
				cj[l] = -b1v*cj1[l] + cj[l]
				cj1[l] = u
			}
			// zOpRotB — s2 >= 0.5 form, on columns 5,6.
			za.rotB(6, b1v, b2)
			for l, cj1, cj := 0, nc(5), nc(6); l < rows; l++ {
				u := b2*cj1[l] + cj[l]
				cj[l] = -cj1[l] + b1v*cj[l]
				cj1[l] = u
			}
			// zOpRotTD — tddiag.f QL rotation, on columns 7,8.
			za.rotTD(7, c, s)
			for k, ci, ci1 := 0, nc(7), nc(8); k < rows; k++ {
				hh := ci1[k]
				ci1[k] = s*ci[k] + c*hh
				ci[k] = c*ci[k] - s*hh
			}
			// zOpScale — the underflow rescale, on column 9.
			za.scale(9, f)
			for l, cj := 0, nc(9); l < rows; l++ {
				cj[l] = f * cj[l]
			}
			za.flush()

			for j := 1; j <= n; j++ {
				col, want := za.col(j), nc(j)
				for r := range col {
					if col[r] != want[r] {
						t.Fatalf("rows=%d workers=%d: column %d row %d = %.17g, naive gave %.17g",
							rows, workers, j, r, col[r], want[r])
					}
				}
			}
		}
	}
}

// BenchmarkBandSymDiagFastRef runs the frozen serial reference at the tier-1 shape, so the
// speedup can be quoted as a ratio measured in one process on one machine rather than against a
// number written down on some other day.
//
// Measure it with -benchtime 3x or more. A single iteration is dominated by first-touch page
// faults on af and z and by a cold cache, which at this shape inflates the first pass by ~3x.
func BenchmarkBandSymDiagFastRef(b *testing.B) {
	_, proto := buildBanded(3712, 37, 7)
	for b.Loop() {
		bs := bandStorage{data: append([]float64(nil), proto.data...), dim: proto.dim, band: proto.band}
		bandSymDiagFastRef(bs)
	}
}

// BenchmarkBandSymDiagFastNarrow is the tier-1 shape as production actually calls it: a `main`
// top strip and a last-block bottom strip rather than the symmetric 2*band. Against
// BenchmarkBandSymDiagFast it shows what decoupling the accumulated rows from the bandwidth is
// worth on its own.
func BenchmarkBandSymDiagFastNarrow(b *testing.B) {
	const dim, band = 3712, 37
	_, proto := buildBanded(dim, band, 7)
	main := (band + 1) / 2
	for b.Loop() {
		bs := bandStorage{data: append([]float64(nil), proto.data...), dim: proto.dim, band: proto.band}
		bandSymDiagFastOpts(bs, bandEigOpts{topRows: main, botRows: main})
	}
}

// BenchmarkZReplayBlock measures the replay's single-core throughput as a function of the row
// block a worker owns, which is what sets zRowMin. The op stream is tddiag's: QL rotations on
// adjacent column pairs. Sub-benchmark name is rows/block.
//
// Measured 2026-09-28 on one core of an EPYC 7513: ~2.0 G updates/s at a 32-row block against
// ~3.3 G at 3080 rows, so a worker given 32 rows runs at ~60% of peak. That is why zRowMin is
// not lowered further — and why the tier-1 {3712,37} shape, whose 74 accumulated rows split into
// two 37-row blocks, shows almost no gain from a second worker even though production's 3080
// rows split into 48-row blocks across 64 of them.
func BenchmarkZReplayBlock(b *testing.B) {
	const nops = 1 << 14
	const n = 4096
	for _, rows := range []int{64, 256, 3080} {
		za := testZAccum(b, n, bandEigOpts{topRows: rows / 2, botRows: rows - rows/2, workers: 1, batch: nops})
		rng := rand.New(rand.NewSource(1))
		for i := range za.z {
			za.z[i] = rng.NormFloat64()
		}
		for o := range nops {
			za.tape.kind[o] = zOpRotTD
			za.tape.col[o] = int32(2 + (o*7)%(n-3))
			za.tape.f1[o] = 0.6
			za.tape.f2[o] = -0.8
		}
		za.tape.n = nops
		for _, blk := range []int{16, 32, 64, 256, 1024, rows} {
			if blk > rows {
				continue
			}
			b.Run(fmt.Sprintf("rows=%d/blk=%d", rows, blk), func(b *testing.B) {
				b.SetBytes(int64(nops) * int64(rows) * 3)
				for b.Loop() {
					for lo := 0; lo < rows; lo += blk {
						za.replay(lo, min(lo+blk, rows))
					}
				}
			})
		}
	}
}

// TestBandEigProgressReportsMonotonically checks the reduction's progress callback: it must fire
// during the reduction, advance monotonically, never exceed the total, and finish with a call at
// col == cols. The last point is what makes a log readable — a stage that stops reporting at 98%
// looks stuck at exactly the moment it succeeded.
func TestBandEigProgressReportsMonotonically(t *testing.T) {
	const dim, band = 4100, 4
	_, bs := buildBanded(dim, band, 5)
	type tick struct {
		col, cols int
		elapsed   time.Duration
	}
	var ticks []tick
	_, _, _, _ = bandSymDiagFastOpts(bs, bandEigOpts{
		topRows: band, botRows: band,
		progress: func(col, cols int, elapsed time.Duration) {
			ticks = append(ticks, tick{col, cols, elapsed})
		},
	})
	if len(ticks) < 3 {
		t.Fatalf("got %d progress calls at dim=%d (every %d columns), want at least 3",
			len(ticks), dim, bandEigProgressEvery)
	}
	for i, tk := range ticks {
		if tk.col < 0 || tk.col > tk.cols || tk.cols <= 0 {
			t.Errorf("tick %d: col=%d cols=%d out of range", i, tk.col, tk.cols)
		}
		if i > 0 && tk.col < ticks[i-1].col {
			t.Errorf("tick %d: col went %d -> %d", i, ticks[i-1].col, tk.col)
		}
		if tk.elapsed < 0 {
			t.Errorf("tick %d: negative elapsed %v", i, tk.elapsed)
		}
	}
	if last := ticks[len(ticks)-1]; last.col != last.cols {
		t.Errorf("final tick is col=%d/%d, want a completion call at col == cols", last.col, last.cols)
	}
}

// TestBandEigBitExactWithParallelBandReplay is the only test that exercises the band-matrix
// replay in parallel, and it is the one that would catch a mistake in aTape's dependency
// argument — that within a chase only the l == maxl column-walk iteration feeds the next step, so
// everything else can be deferred and replayed in any order.
//
// It needs the grain override. A chase carries ~4*(dim-k) band-matrix updates, so at the default
// grain of 4096 even the {3712,37} case splits nothing: every chase runs on the caller and the
// parallel path is never entered. Forcing a grain of 8 splits the same chases across as many
// workers as the pool has, at shapes a test can run in milliseconds.
//
// The failure mode this guards against is a silently wrong band-matrix element, which propagates
// into an eigenvalue rather than crashing, so the comparison is against the frozen serial
// reference and it is on bits.
func TestBandEigBitExactWithParallelBandReplay(t *testing.T) {
	for _, tc := range []struct{ dim, band int }{{200, 2}, {260, 128}, {512, 5}, {1000, 10}} {
		seed := int64(1000 + tc.dim*100 + tc.band)
		_, ref := buildBanded(tc.dim, tc.band, seed)
		wantD, wantZ := bandSymDiagFastRef(ref)
		// grain 1 is a barrier per single update — the harshest ordering test, but too slow to run
		// on every shape, so it is kept for the cheapest one.
		grains := []int{8, 37, 4096}
		if tc.dim <= 256 {
			grains = append(grains, 1)
		}
		for _, grain := range grains {
			for _, workers := range []int{2, 5, 0} {
				_, bs := buildBanded(tc.dim, tc.band, seed)
				gotD, gotZ, ld, _ := bandSymDiagFastOpts(bs, bandEigOpts{
					topRows: tc.band, botRows: tc.band, workers: workers, aGrain: grain,
				})
				tag := fmt.Sprintf("dim=%d band=%d aGrain=%d workers=%d", tc.dim, tc.band, grain, workers)
				bandEigCmpRef(t, tag, tc.dim, tc.band, ld, gotD, gotZ, wantD, wantZ)
			}
		}
	}
}

// TestZOpKindsMatchDevice pins the op numbering that backend/bandeig_kernels.cu hard-codes as
// Z_OP_ROT_A..Z_OP_SCALE. The device kernel switches on these values, so renumbering them here
// would silently apply the wrong rotation on the GPU — a wrong eigenvector row, not a crash, and
// invisible to every CPU test. Change one side and this fails; change both and update the comment.
func TestZOpKindsMatchDevice(t *testing.T) {
	for _, tc := range []struct {
		got  uint8
		want uint8
		name string
	}{
		{zOpRotA, 0, "Z_OP_ROT_A"},
		{zOpRotB, 1, "Z_OP_ROT_B"},
		{zOpRotTD, 2, "Z_OP_ROT_TD"},
		{zOpScale, 3, "Z_OP_SCALE"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s is %d here, but backend/bandeig_kernels.cu defines it as %d",
				tc.name, tc.got, tc.want)
		}
	}
}

// stripOrthErr returns max |S·Sᵀ - I| over the accumulated strip, where S[r][k] = z[k*ld+r].
//
// This is the one check on the eigenvector strip that needs no reference implementation at all.
// S = Eᵀ·Q for the selection E of the retained rows and the orthogonal Q that diagonalizes the band
// matrix, so S·Sᵀ = Eᵀ·Q·Qᵀ·E = Eᵀ·E = I whenever the retained rows are distinct — whatever
// algorithm produced Q. It therefore catches a dropped rotation, a replay in the wrong order, an
// operand swapped in a kernel, a transform applied to the wrong column range, and a stale host copy,
// none of which crash and all of which would otherwise surface only as a wrong pole strength.
//
// What it cannot catch is an orthogonal error — a permutation or a sign flip of whole eigenvectors —
// which is exactly the class that is physically harmless here, since signs are gauge.
func stripOrthErr(z []float64, rows, ld, dim int) float64 {
	worst := 0.0
	for r := range rows {
		for rp := r; rp < rows; rp++ {
			var acc float64
			for k := range dim {
				acc += z[k*ld+r] * z[k*ld+rp]
			}
			want := 0.0
			if r == rp {
				want = 1.0
			}
			if e := math.Abs(acc - want); e > worst {
				worst = e
			}
		}
	}
	return worst
}

// TestBandEigStripIsOrthonormal pins the invariant above on the current solver, across the shapes
// the rest of the suite uses. It is the validation backbone for any future change to how the strip
// is accumulated: a replacement algorithm cannot be bit-compared against the reference, but it must
// still satisfy this.
//
// The two strips have to be disjoint for the identity to hold, which is the production case and what
// diagProjected already guarantees by falling back to a dense solve when main+lastSize > dim.
func TestBandEigStripIsOrthonormal(t *testing.T) {
	for _, tc := range []struct{ dim, band int }{
		{12, 2}, {30, 5}, {40, 7}, {50, 12}, {200, 2}, {260, 128},
	} {
		main := (tc.band + 1) / 2
		if 2*main > tc.dim {
			continue
		}
		_, bs := buildBanded(tc.dim, tc.band, int64(4000+tc.dim))
		_, z, ld, _ := bandSymDiagFastOpts(bs, bandEigOpts{topRows: main, botRows: main})
		if got := stripOrthErr(z, 2*main, ld, tc.dim); got > 1e-12 {
			t.Errorf("dim=%d band=%d: max|S·Sᵀ-I| = %.3e, want <= 1e-12", tc.dim, tc.band, got)
		}
	}
}

// TestZTapeSegmentsCommute is the guard on the claim that makes a parallel device replay possible:
// within a segment, the recorded ops touch pairwise-disjoint column pairs, so they commute and can be
// applied concurrently rather than one after another.
//
// It matters because the device kernel's comment used to assert the opposite ("no more parallelism
// available than rows, because a row's ops are strictly ordered"), which is true of tddiag's chained
// QL rotations and false of bnd2td's chase — where j advances by m1 >= 2 and each step rotates columns
// (j-1, j). Believing the wrong one is what left the kernel at 4% of HBM bandwidth.
//
// The test drives the real reduction and inspects the real tape, rather than reasoning about the loop,
// because the whole point is that the boundaries are recorded correctly by bnd2td and not just true in
// principle.
func TestZTapeSegmentsCommute(t *testing.T) {
	// {512,5} and {1000,10} are here because they are the shapes that FAILED on the device, and they
	// failed for a reason none of the smaller shapes could expose: dim > 64 with a narrow band makes
	// the periodic underflow rescale fire (549 times at {512,5}), and its scale ops were being merged
	// into a run with the next chase's rotations, which act on the same columns. rescaleFires below
	// asserts the coverage rather than assuming it, since a rescale that stops firing would make these
	// cases quietly equivalent to the others.
	for _, tc := range []struct {
		dim, band    int
		rescaleFires bool
	}{{40, 7, false}, {60, 12, false}, {200, 20, false}, {260, 40, false}, {512, 5, true}, {1000, 10, true}} {
		main := (tc.band + 1) / 2
		_, bs := buildBanded(tc.dim, tc.band, int64(6100+tc.dim))
		dim := bs.dim
		b1 := bs.band + 1

		pool := parallel.NewFixedPool(1)
		o := bandEigOpts{topRows: main, botRows: main, workers: 1, batch: 1 << 20}
		za := newZAccum(dim, o, pool)
		af := packBandColumnMajor(bs)
		d := make([]float64, dim)
		e := make([]float64, dim)
		e2 := make([]float64, dim)

		// Reach into the reduction just far enough to inspect a tape that has segments on it: run it
		// with a batch large enough that nothing auto-flushes, then examine what accumulated.
		bnd2td(za, dim, b1, af, d, e, e2, pool, 0, nil, nil, 0, nil, 0)
		segs := za.segments()
		pool.Close()

		if len(segs) < 2 {
			t.Errorf("dim=%d band=%d: %d segments recorded; the chase boundaries are not being marked",
				tc.dim, tc.band, len(segs))
			continue
		}
		scales := 0
		for o := range za.tape.n {
			if za.tape.kind[o] == zOpScale {
				scales++
			}
		}
		// scaleAll at label 800 always records dim-1 of them, so only the excess is the rescale's.
		if rescaled := scales - (dim - 1); tc.rescaleFires && rescaled <= 0 {
			t.Errorf("dim=%d band=%d: %d scale ops, none beyond scaleAll's %d, so the underflow rescale "+
				"never fired and this case no longer covers the run boundary it was added for",
				tc.dim, tc.band, scales, dim-1)
		}

		multi := 0
		for _, s := range segs {
			if !s.par {
				continue // a sequential run needs no disjointness; only a concurrent claim does
			}
			seen := map[int32]bool{}
			for o := s.lo; o < s.hi; o++ {
				// Every op kind touches its column and one neighbour: col-1 for the bnd2td rotations,
				// col+1 for the QL ones, and col alone for a scale.
				cols := []int32{za.tape.col[o]}
				switch za.tape.kind[o] {
				case zOpRotA, zOpRotB:
					cols = append(cols, za.tape.col[o]-1)
				case zOpRotTD:
					cols = append(cols, za.tape.col[o]+1)
				}
				for _, c := range cols {
					if seen[c] {
						t.Fatalf("dim=%d band=%d: run [%d,%d) claims its ops commute but touches column %d "+
							"twice — a concurrent replay would be wrong",
							tc.dim, tc.band, s.lo, s.hi, c)
					}
					seen[c] = true
				}
			}
			if s.hi-s.lo > 1 {
				multi++
			}
		}
		if multi == 0 {
			t.Errorf("dim=%d band=%d: no concurrent run holds more than one op, so there is no "+
				"parallelism to gain", tc.dim, tc.band)
		}
		t.Logf("dim=%d band=%d: %d runs, %d concurrent with >1 op, %d rescale ops",
			tc.dim, tc.band, len(segs), multi, scales-(dim-1))
	}
}

// TestConcurrentRunOpsCommute proves the property the device's 2-D replay rests on, rather than
// asserting it: applying a concurrent run's ops in REVERSE order must give a bit-identical strip.
//
// Disjointness (TestZTapeSegmentsCommute) says no two ops in a run touch the same column, which
// implies commutativity; this checks the implication directly and on bits, because a concurrent kernel
// effectively picks an arbitrary order and "bit-identical under reordering" is exactly the licence it
// needs. If this ever fails, the 2-D kernel is unsound no matter what the disjointness check says.
func TestConcurrentRunOpsCommute(t *testing.T) {
	const rows, ld, n = 64, 64, 4096
	rng := rand.New(rand.NewSource(4242))
	base := make([]float64, ld*n)
	for i := range base {
		base[i] = rng.NormFloat64()
	}
	// A run shaped like the band reduction's: rotations on (j-1, j) with j advancing by a stride > 1,
	// so the column pairs are pairwise disjoint.
	const stride, nops = 7, 400
	kind := make([]uint8, nops)
	col := make([]int32, nops)
	f1 := make([]float64, nops)
	f2 := make([]float64, nops)
	for o := range nops {
		kind[o] = zOpRotA
		if o%3 == 1 {
			kind[o] = zOpRotB
		}
		col[o] = int32(2 + o*stride)
		f1[o], f2[o] = rng.NormFloat64(), rng.NormFloat64()
	}

	run := func(order []int) []float64 {
		pool := parallel.NewFixedPool(1)
		defer pool.Close()
		za := newZAccum(n, bandEigOpts{topRows: rows / 2, botRows: rows / 2, workers: 1, batch: nops}, pool)
		copy(za.z, base)
		for _, o := range order {
			za.tape.kind[za.tape.n] = kind[o]
			za.tape.col[za.tape.n] = col[o]
			za.tape.f1[za.tape.n] = f1[o]
			za.tape.f2[za.tape.n] = f2[o]
			za.tape.n++
		}
		za.replay(0, za.rows)
		return za.z
	}

	fwd := make([]int, nops)
	for i := range fwd {
		fwd[i] = i
	}
	rev := make([]int, nops)
	for i := range rev {
		rev[i] = nops - 1 - i
	}
	a, b := run(fwd), run(rev)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("z[%d] = %.17g forward, %.17g reversed — the run's ops do NOT commute, so the "+
				"concurrent device kernel would be unsound", i, a[i], b[i])
		}
	}

	// Control: a run that SHARES a column must NOT be order-independent, or the test above proves
	// nothing about the disjointness precondition.
	col[1] = col[0]
	if c, d := run(fwd), run(rev); bytesEqual(c, d) {
		t.Error("a run with two ops on the same column came out order-independent; the test cannot " +
			"distinguish commuting from non-commuting runs")
	}
}

func bytesEqual(a, b []float64) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
