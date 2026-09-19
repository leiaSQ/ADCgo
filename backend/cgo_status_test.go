package backend

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestCgoStatusReturnsAreChecked fails when a cgo shim that returns a status code is called as a
// bare statement, discarding it.
//
// This is the mechanical guard for the class of bug that made cudaError_t 719 unattributable for
// weeks. All five kernel launchers return cudaGetLastError(), and every call site discarded it;
// dev_free's shim returned void outright. A launch that failed therefore said nothing, and the
// sticky error only surfaced later at an unrelated cudaDeviceSynchronize — which is why the DIP
// crash was always reported from distBackend.gemmMatOne, where it was never raised (job 14561251).
//
// It runs WITHOUT the cuda build tag on purpose: it parses the tagged files as source text, so a
// machine with no GPU and no CUDA toolkit still enforces the rule. That is the whole point — this
// is the cheap gate that runs on every commit, ahead of the hardware smoke.
//
// The rule: a call to C.<fn> used as an expression STATEMENT discards whatever it returns. That is
// only acceptable when the shim is declared void, so the void ones are listed explicitly. Adding a
// new status-returning shim and ignoring it will fail here; adding a void one means adding it to
// the list, which is a deliberate act rather than an oversight.
func TestCgoStatusReturnsAreChecked(t *testing.T) {
	// Shims declared `static void` in the cgo preamble of cuda.go / hip.go. Everything else
	// returns a status (cudaError_t, cublasStatus_t, a count, or a pointer) and must be used.
	voidShims := map[string]bool{
		"blas_axpy": true, "blas_gemv": true, "blas_scal": true,
		"dev_clear_error": true, "solver_destroy": true,
		// hip.go's shims are void where cuda.go's return a status; both files are scanned, so
		// the union is listed and each file is checked against its own preamble below.
		"dev_free": true, "dev_zero": true, "dev_h2d": true, "dev_d2h": true,
		"dev_d2d": true, "dev_sync": true,
	}

	files, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read backend dir: %v", err)
	}
	scanned := 0
	for _, f := range files {
		name := f.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if !strings.Contains(string(src), "\"C\"") {
			continue // not a cgo file
		}
		// Which shims does THIS file declare void? A name void here may return a status in the
		// sibling backend, so trust the file's own preamble over the union above.
		voidHere := map[string]bool{}
		for _, m := range regexp.MustCompile(`static\s+void\s+([a-z_0-9]+)\s*\(`).FindAllStringSubmatch(string(src), -1) {
			voidHere[m[1]] = true
		}

		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, name, src, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		scanned++

		ast.Inspect(file, func(n ast.Node) bool {
			stmt, ok := n.(*ast.ExprStmt) // a call used as a statement => return value discarded
			if !ok {
				return true
			}
			call, ok := stmt.X.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			pkg, ok := sel.X.(*ast.Ident)
			if !ok || pkg.Name != "C" {
				return true
			}
			fn := sel.Sel.Name
			if voidHere[fn] || (len(voidHere) == 0 && voidShims[fn]) {
				return true
			}
			t.Errorf("%s:%d: C.%s(...) is called as a statement, discarding its return value.\n"+
				"\tIf it returns a status, check it (ckLaunch for kernel launchers, ckCuda/ckBlas "+
				"otherwise) — an ignored status is why a failed launch stayed silent until an "+
				"unrelated sync reported it.\n"+
				"\tIf the shim really is `static void`, it will be picked up from this file's own "+
				"preamble; declare it void there rather than adding an exception here.",
				name, fset.Position(stmt.Pos()).Line, fn)
			return true
		})
	}
	if scanned == 0 {
		t.Fatal("no cgo files scanned — the guard would pass vacuously; has the build layout changed?")
	}
	t.Logf("scanned %d cgo file(s) for discarded status returns", scanned)
}
