package main

import (
	"fmt"
	"os"

	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
)

// buildSigma resolves -sigma into the static self-energy the SIP main block subtracts.
//
// The ADC matrix code does not build Σ: theADCcode keeps it in a separate module
// (`&self-energy`) and its ndadc3ip subtracts the result (build_main_block ends with
// `main_block->daxpy(-1., *sigma_)`), while the ADC(4) core takes Σ as an input to adc_().
// Leaving it out shifts every main line by ~0.2–0.35 eV; satellites are unaffected.
//
// "auto" is Σ(∞): the all-order resolvent resummation, which ADCgo reproduces bit-exactly
// against theADCcode. It is the scheme that distinguishes this code from implementations that
// truncate the static self-energy perturbatively; three/four/fplus remain available for
// comparison. -sigma-akrit / -sigma-maxit tune the resolvent iteration (theADCcode's own
// defaults are 1e-9 / 30; tighter values converge to the exact fixed point).
func buildSigma(cfg sipConfig, ints *integrals.Store, eps []float64, nocc, norb int) (func(i, j int) float64, error) {
	name := cfg.sigma
	switch name {
	case "off":
		return nil, nil
	case "auto", "":
		// The all-order Σ(∞) is what theADCcode itself uses and what the ADC(4)/CVS reference
		// tapes were generated with; ADCgo reproduces it bit-exactly. The perturbative schemes
		// remain selectable for comparison.
		name = "infinite"
	}

	scheme, err := selfenergy.ParseScheme(name)
	if err != nil {
		return nil, err
	}
	opts := selfenergy.Options{Akrit: cfg.sigmaAkrit, MaxIt: cfg.sigmaMaxIt}
	sig, err := selfenergy.Static(ints, eps, nocc, norb, scheme, opts)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "static self-energy: %v\n", scheme)
	return sig.Func(), nil
}
