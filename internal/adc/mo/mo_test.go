package mo

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
)

func testdata(name string) string { return filepath.Join("..", "..", "..", "testdata", name) }

func load(t *testing.T, name string) *Data {
	t.Helper()
	d, err := ReadFile(testdata(name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return d
}

// writeJSON dumps doc to a temp file and returns its path.
func writeJSON(t *testing.T, doc map[string]any) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "side.json")
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// legacyDoc is a sidecar as written before the dipole keys existed: a 1-AO, 1-MO
// molecule, enough to exercise the decoder.
func legacyDoc() map[string]any {
	return map[string]any{
		"nao":        1,
		"nmo":        1,
		"mo_coeff":   [][]float64{{1}},
		"overlap":    [][]float64{{1}},
		"ao_atom":    []int{0},
		"atom_names": []string{"H"},
	}
}

// TestReadFileLegacySidecar: sidecars written before the dipole keys existed must
// still load. This is the backward-compatibility contract the new keys rest on.
func TestReadFileLegacySidecar(t *testing.T) {
	d, err := ReadFile(writeJSON(t, legacyDoc()))
	if err != nil {
		t.Fatalf("legacy sidecar failed to load: %v", err)
	}
	if d.HasDipole {
		t.Error("HasDipole set for a sidecar with no dipole keys")
	}
}

// TestReadFilePartialDipoleIsError: half a dipole block is a broken dumper, and it must
// surface as a parse error rather than as a wrong transition moment much later.
func TestReadFilePartialDipoleIsError(t *testing.T) {
	doc := legacyDoc()
	doc["dip_ao"] = [][][]float64{{{0}}, {{0}}, {{0}}}
	doc["dip_origin"] = []float64{0, 0, 0}
	// atom_coords and atom_charges deliberately absent.
	if _, err := ReadFile(writeJSON(t, doc)); err == nil {
		t.Fatal("partial dipole key set loaded without error")
	}
}

// TestMOOrthonormality: Cᵀ·S·C = I. The overlap has been in the sidecar all along, so
// this validates Transpose and TransformMO against data nothing else in the pipeline
// could have massaged.
func TestMOOrthonormality(t *testing.T) {
	d := load(t, "h2o.mo.json")
	got := TransformMO(d.S, d.C)
	var maxErr float64
	for i := range d.NMO {
		for j := range d.NMO {
			want := 0.0
			if i == j {
				want = 1.0
			}
			if e := math.Abs(got.At(i, j) - want); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxErr > 1e-10 {
		t.Errorf("‖CᵀSC − I‖_max = %g", maxErr)
	}
}

func TestDipMOSymmetric(t *testing.T) {
	d := load(t, "h2o.mo.json")
	if !d.HasDipole {
		t.Fatal("h2o.mo.json has no dipole keys")
	}
	for x := range 3 {
		for i := range d.NMO {
			for j := range i {
				if e := math.Abs(d.DipMO[x].At(i, j) - d.DipMO[x].At(j, i)); e > 1e-12 {
					t.Fatalf("DipMO[%d] asymmetric at (%d,%d) by %g", x, i, j, e)
				}
			}
		}
	}
}

// TestGroundStateDipole reconstructs the SCF dipole from the dumped AO integrals, the
// MO coefficients and the geometry, and compares it to the value pyscf computed for
// itself. It is the gate on the electron-charge sign, the orientation of C in the
// AO→MO transform, and the gauge origin — each of which fails silently on its own.
func TestGroundStateDipole(t *testing.T) {
	d := load(t, "h2o.mo.json")
	if !d.HasDipole {
		t.Fatal("h2o.mo.json has no dipole keys")
	}
	fd, err := fcidump.ReadFile(testdata("h2o.fcidump"))
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	if d.NMO != fd.NORB {
		t.Fatalf("sidecar has %d MOs, fcidump %d: this gate needs the full MO space",
			d.NMO, fd.NORB)
	}
	got := d.GroundStateDipole(fd.NELEC / 2)
	for x := range 3 {
		if e := math.Abs(got[x] - d.SCFDip[x]); e > 1e-8 {
			t.Errorf("dipole component %d: got %.12f, pyscf scf_dip %.12f (Δ=%.2e)",
				x, got[x], d.SCFDip[x], e)
		}
	}
}

// TestFrozenCoreSidecarLoads: the DZP reference dumps only the active MOs. It must load
// and expose a full-dimension DipAO with an active-window DipMO.
func TestFrozenCoreSidecarLoads(t *testing.T) {
	d := load(t, "h2o_dzp.mo.json")
	if !d.HasDipole {
		t.Fatal("h2o_dzp.mo.json has no dipole keys")
	}
	if d.DipAO[0].Rows != d.NAO || d.DipMO[0].Rows != d.NMO {
		t.Errorf("DipAO is %d² (want nao=%d), DipMO is %d² (want nmo=%d)",
			d.DipAO[0].Rows, d.NAO, d.DipMO[0].Rows, d.NMO)
	}
	if len(d.AtomCharges) != len(d.AtomNames) {
		t.Errorf("%d atom charges but %d atom names", len(d.AtomCharges), len(d.AtomNames))
	}
}

// TestLabelledSidecar: the orbital-label group of a localized-orbital dump
// (testdata/khci/he3_ghost.in via scripts/fcidump/dump_fcidump.py) loads, is consistent, and
// the rotated MOs are orthonormal.
func TestLabelledSidecar(t *testing.T) {
	d := load(t, filepath.Join("khci", "he3_ghost.mo.json"))
	if !d.HasLabels || d.Canonical {
		t.Fatalf("HasLabels=%v Canonical=%v, want true/false", d.HasLabels, d.Canonical)
	}
	if n := d.NOccLabelled(); n != 3 {
		t.Fatalf("%d occupied labels, want 3", n)
	}
	ghosts := 0
	for a, g := range d.GhostAtom {
		if g {
			ghosts++
			if d.AtomNames[a] != "GT" && d.AtomNames[a] != "GOFF" {
				t.Errorf("ghost %d named %q", a, d.AtomNames[a])
			}
		}
	}
	if ghosts != 2 {
		t.Errorf("%d ghost centres, want 2", ghosts)
	}
	for m, k := range d.OrbKind {
		if (k == OrbFree) != (d.OrbAtom[m] == -1) {
			t.Errorf("MO %d kind %v with atom %d", m, k, d.OrbAtom[m])
		}
	}
	// C^T S C = 1 for the rotated basis
	var worst float64
	for p := range d.NMO {
		for q := range d.NMO {
			var v float64
			for i := range d.NAO {
				for j := range d.NAO {
					v += d.C.At(i, p) * d.S.At(i, j) * d.C.At(j, q)
				}
			}
			if p == q {
				v--
			}
			worst = max(worst, math.Abs(v))
		}
	}
	if worst > 1e-10 {
		t.Errorf("rotated MOs not orthonormal: %g", worst)
	}
}

// TestReadFilePartialLabelsIsError: the label group is all or none.
func TestReadFilePartialLabelsIsError(t *testing.T) {
	doc := legacyDoc()
	doc["orb_kind"] = []string{"occ"}
	if _, err := ReadFile(writeJSON(t, doc)); err == nil {
		t.Fatal("a sidecar with 1 of the 4 label keys loaded")
	}
	doc["orb_atom"] = []int{0}
	doc["ghost_atoms"] = []bool{true}
	doc["canonical"] = false
	if _, err := ReadFile(writeJSON(t, doc)); err == nil {
		t.Fatal("an occupied orbital assigned to a ghost loaded")
	}
}
