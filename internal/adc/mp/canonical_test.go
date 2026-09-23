package mp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
)

// TestRequireCanonical: every canonical FCIDUMP in testdata passes, and the
// localized-orbital dumps (rotated on purpose) are refused.
func TestRequireCanonical(t *testing.T) {
	root := filepath.Join("..", "..", "..", "testdata")
	var paths []string
	filepath.WalkDir(root, func(p string, e os.DirEntry, err error) error {
		if err == nil && !e.IsDir() && strings.HasSuffix(p, ".fcidump") {
			paths = append(paths, p)
		}
		return nil
	})
	if len(paths) == 0 {
		t.Fatal("no FCIDUMPs found")
	}
	sawLocalized := false
	for _, p := range paths {
		d, err := fcidump.ReadFile(p)
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		m := MaxFockOffDiagonal(d, NOcc(d))
		if strings.Contains(p, "matched") {
			// theADCcode's matched-integral tape is semi-canonical by construction (see
			// RequireCanonical); it is not a canonical-HF dump and not a localized one
			continue
		}
		localized := strings.Contains(p, "localized") || strings.Contains(p, "he3_ghost")
		sawLocalized = sawLocalized || localized
		err = RequireCanonical(d, NOcc(d))
		t.Logf("%s: max off-diagonal Fock %.2e", filepath.Base(p), m)
		if localized && err == nil {
			t.Errorf("%s: localized dump accepted as canonical (max off-diagonal %.2e)", p, m)
		}
		if !localized && err != nil {
			t.Errorf("%s: %v", p, err)
		}
	}
	if !sawLocalized {
		t.Error("no localized dump in testdata: the refusal path is untested")
	}
}
