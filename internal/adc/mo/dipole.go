package mo

import "github.com/leiaSQ/ADCgo/internal/adc/backend"

// TransformMO carries a one-electron AO-basis matrix into the MO basis: Cᵀ·A·C, with
// A nAO × nAO and C nAO × nMO. The result is nMO × nMO over exactly the MOs the
// sidecar carries, which for a frozen-core dump is the active window.
func TransformMO(a, c backend.Mat) backend.Mat {
	return backend.MatMul(backend.MatMul(backend.Transpose(c), a), c)
}

// NuclearDipole is Σ_A Z_A·(R_A − origin), the nuclear half of the dipole about the
// sidecar's gauge origin (a.u.).
func (d *Data) NuclearDipole() [3]float64 {
	var mu [3]float64
	for a, z := range d.AtomCharges {
		for x := range 3 {
			mu[x] += z * (d.AtomCoords[a][x] - d.DipOrigin[x])
		}
	}
	return mu
}

// GroundStateDipole is the closed-shell SCF dipole about the sidecar's gauge origin:
// the nuclear term minus the electronic one, 2·Σ_{i<nocc} ⟨i|r|i⟩. The minus sign is
// the electron's charge — the same flip RASSI applies to its MLTPL integrals
// (mk_prop.F90:107-109) — and it is the one convention here that a test has to pin
// down, since getting it backwards still produces a plausible-looking vector.
//
// It reproduces the dumper's SCFDip only when C spans every occupied orbital. A
// frozen-core sidecar has no core columns to trace over, so its electronic term is
// short by the core contribution.
func (d *Data) GroundStateDipole(nocc int) [3]float64 {
	mu := d.NuclearDipole()
	for x := range 3 {
		for i := range nocc {
			mu[x] -= 2 * d.DipMO[x].At(i, i)
		}
	}
	return mu
}
