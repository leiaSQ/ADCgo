// Temporary throwaway: exact DIP sector sizing (n, main, block-sparse operator) for one or
// more FCIDUMPs, via the same functions the solver's pre-flight guard uses. CPU-only.
// Prints n/main first (cheap), then the operator (a full host block walk), then the
// whole-band single-GPU need and the per-GPU footprint for a range of -mgpu partition counts.
package main

import (
	"fmt"
	"os"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

const GB = 1 << 30

func main() {
	for _, path := range os.Args[1:] {
		probe(path)
	}
}

func probe(path string) {
	d, err := fcidump.ReadFile(path)
	if err != nil {
		panic(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	orbSym := d.OrbSym
	ints := integrals.New(d, nocc, orbSym)
	be, err := backend.New("gonum")
	if err != nil {
		panic(err)
	}
	fmt.Printf("\n==== %s  NORB=%d NELEC=%d nocc=%d nvir=%d ====\n", path, d.NORB, d.NELEC, nocc, d.NORB-nocc)

	names := map[dip.Spin]string{dip.Singlet: "singlet", dip.Triplet: "triplet"}
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		sp := dip.NewSpace(nocc, d.NORB, orbSym, 0, spin)
		n, main := sp.Size(), sp.MainBlockSize()
		panels := 4 * uint64(n) * uint64(main) * 8
		fmt.Printf("\nspin=%s  n=%d  main=%d  panels(4*n*main*8)=%.1f GB   [walking operator...]\n",
			names[spin], n, main, float64(panels)/GB)
		os.Stdout.Sync()
		mx := dip.New(sp, ints, eps, be)
		op := mx.OperatorResidentBytes()
		mx.Release()
		need := panels + op
		fmt.Printf("spin=%s  operator=%.1f GB  ->  whole-band single-GPU need=%.1f GB\n",
			names[spin], float64(op)/GB, float64(need)/GB)
		fmt.Printf("   per-GPU under -mgpu N (need/N, H200=141 GB):\n")
		for _, N := range []int{1, 2, 4, 6, 8} {
			fmt.Printf("     mgpu %-2d : %7.1f GB/GPU  %s\n", N, float64(need)/GB/float64(N),
				fitTag(float64(need)/GB/float64(N)))
		}
	}
}

func fitTag(perGPU float64) string {
	switch {
	case perGPU <= 120:
		return "fits (comfortable)"
	case perGPU <= 138:
		return "fits (tight)"
	default:
		return "DOES NOT FIT"
	}
}
