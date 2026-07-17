// bandeig.go — a banded real-symmetric eigensolver that returns, for each eigenvalue,
// only the top `band` and bottom `band` rows of its eigenvector. It is a Go port of
// Tarantelli's bnd2td + tddiag (../ADC/libLanczos/{bnd2td,tddiag}.f, wrapped by
// band_sym_diag_fast in lanczos_util.cpp).
//
// Why not LAPACK: the short-recurrence low-memory Lanczos driver (lowmem.go, Mode B)
// discards the Krylov basis, so it cannot back-transform full eigenvectors. But it does
// not need them: the first `band` Lanczos vectors are the main-space unit vectors, so the
// top `band` components of a projected eigenvector ARE its main-space components (its pole
// strength), and the bottom `band` are the residual-tail slice. bnd2td/tddiag accumulate
// exactly and only those 2*band rows through the band→tridiagonal→QL rotations, making the
// eigenvector cost O(dim*band) instead of the O(dim*dim) a full LAPACK dsbev would spend —
// which for the melanin block width (band ≈ 1700, dim ≈ 10^5) is the difference between a
// few GB and ~900 GB. The eigenvalue reduction itself is O(dim^2 * band); it runs once per
// sector solve, off the mat-vec hot path.
//
// The port keeps the reference's Fortran control flow (including the underflow rescale and
// the arithmetic-IF band-width dispatch) line for line, so it can be checked against the
// Fortran; the indexing helpers below translate the 1-based column-major a(n,mb) / z(nm,n)
// storage into flat Go slices.
package lanczos

import "math"

// bandStorage holds the projected banded matrix in the short-recurrence driver's native
// layout: dim columns, each of band+1 entries. col j occupies data[j*(band+1) : (j+1)*(band+1)],
// with entry 0 the diagonal and entry i (1..band) the coupling between Lanczos vectors j
// and j+i (the i-th sub-diagonal). This is exactly the `subdiags[col][0..band]` accumulation
// of theADCcode's LanczosEngine and of lowmem.go.
type bandStorage struct {
	data []float64 // dim*(band+1)
	dim  int
	band int
}

// newBandStorage allocates a zeroed banded matrix for dim Lanczos vectors and half-bandwidth
// band (= the block width).
func newBandStorage(dim, band int) bandStorage {
	return bandStorage{data: make([]float64, dim*(band+1)), dim: dim, band: band}
}

// set stores the i-th band entry (i=0 diagonal) of column col.
func (b bandStorage) set(col, i int, v float64) { b.data[col*(b.band+1)+i] = v }

// at returns the i-th band entry (i=0 diagonal) of column col.
func (b bandStorage) at(col, i int) float64 { return b.data[col*(b.band+1)+i] }

// bandSymDiagFast diagonalizes the banded matrix bs and returns the ascending eigenvalues
// (length dim) together with a (2*band)×dim column-major slice z: column i is eigenvector
// i, rows [0,band) its top (main-space) components, rows [band,2*band) its bottom
// (residual-tail) components. Port of band_sym_diag_fast (lanczos_util.cpp:105).
//
// For band == 0 (no satellites) it is a plain diagonal read; z is then the 0×dim empty
// slice and every eigenvalue is its own diagonal entry.
func bandSymDiagFast(bs bandStorage) (evals []float64, z []float64) {
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

	bnd2td(nm, dim, b1, af, d, e, e2, zf)
	tddiag(nm, dim, d, e, zf)

	return d, zf
}

// bnd2td reduces the real-symmetric band matrix a(n,mb) (mb = band+1 stored diagonals,
// column-major, diagonal in the last band-column) to tridiagonal form (d, e, e2), while
// accumulating into z(nm,n) only the rows that start as identity on the first nm/2 and last
// nm/2 columns. Port of ../ADC/libLanczos/bnd2td.f. All indices below mirror the Fortran
// 1-based arithmetic; A/Z access the flat slices column-major.
func bnd2td(nm, n, mb int, a, d, e, e2, z []float64) {
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
			// Periodic underflow rescale every 64 columns (bnd2td.f:124-145).
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

// tddiag is the QL-with-implicit-shifts tridiagonal eigensolver (modified EISPACK tql2),
// applying the same rotations to the nm tracked rows of z. On exit d holds the ascending
// eigenvalues and z(nm,n) the correspondingly permuted partial eigenvectors. Port of
// ../ADC/libLanczos/tddiag.f. Returns the index of a non-converged root, or 0 on success.
func tddiag(nm, n int, d, e, z []float64) int {
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
