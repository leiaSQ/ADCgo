// Command sigmadump prints the static self-energy Σ(0,0) (the CVS core-hole
// diagonal) computed by every scheme, for comparison against theADCcode's
// SIGMA_STATIC / COSTANTI value.
//
//	go run ./cmd/sigmadump <file.fcidump>
package main

import (
	"fmt"
	"os"

	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
)

const au2eV = 27.211396

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: sigmadump <file.fcidump>")
		os.Exit(2)
	}
	d, err := fcidump.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "read fcidump:", err)
		os.Exit(1)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	fmt.Printf("NORB=%d nocc=%d  core orbital = active index 0 (ε=%.6f Ha)\n", d.NORB, nocc, eps[0])
	fmt.Printf("legacy reference (NSOB.N1s.res / EGF): Σ(0,0) = 0.03439610 Ha = 0.935973 eV\n")
	fmt.Printf("(legacy COSTANTI cst.x used IORDER=3, MAXORB=103, AKRIT=1e-6, MAXIT=40)\n\n")

	schemes := []struct {
		name string
		s    selfenergy.Scheme
	}{
		{"three", selfenergy.Three},
		{"four", selfenergy.Four},
		{"fplus", selfenergy.FourPlus},
		{"infinite", selfenergy.Infinite},
	}
	// Second arg "infinite" restricts to the all-order scheme (skips the O(N^6) perturbative
	// four/fplus), which is the direct comparison against COSTANTI's resolvent Σ.
	if len(os.Args) > 2 && os.Args[2] == "infinite" {
		schemes = schemes[3:]
	}
	optsets := []struct {
		label string
		o     selfenergy.Options
	}{
		{"theADCcode (Akrit=1e-9,MaxIt=30)", selfenergy.TheADCcodeDefaults},
		{"tight (ADCgo -sigma auto default)", selfenergy.Options{}},
	}
	for _, op := range optsets {
		fmt.Printf("== opts: %s ==\n", op.label)
		for _, sc := range schemes {
			fmt.Fprintf(os.Stderr, "computing %s / %s ...\n", op.label, sc.name)
			sig, err := selfenergy.Static(ints, eps, nocc, d.NORB, sc.s, op.o)
			if err != nil {
				fmt.Printf("  %-9s ERROR %v\n", sc.name, err)
				continue
			}
			v := sig.At(0, 0)
			fmt.Printf("  %-9s Σ(0,0) = %.8f Ha = %.6f eV   (Δ vs ref = %+.6f eV)\n",
				sc.name, v, v*au2eV, v*au2eV-0.935973)
			// the resulting bare 1h diagonal (ionization) for context
			fmt.Printf("            -ε-Σ = %.6f eV\n", (-eps[0]-v)*au2eV)
		}
		fmt.Println()
	}
}
