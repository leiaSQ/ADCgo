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

	// Σ is by far the most expensive phase of a large SIP run (78 h for the production system) and is only n²
	// floats, so it is cached across processes — see sigma_cache.go for the guard and the
	// motivation. A cache miss, a corrupt file or a stale key all just rebuild.
	cachePath := cfg.sigmaCache
	if cachePath == "auto" || cachePath == "" {
		cachePath = cfg.fcidumpPath + ".sigma-" + name + ".cache"
	}
	key := sigmaCacheKeyFor(name, norb, nocc, opts.MaxIt, opts.Akrit, eps, cfg.fcidumpPath)
	if cachePath != "off" {
		switch cached, err := readSigmaCache(cachePath, key); {
		case err != nil:
			fmt.Fprintf(os.Stderr, "static self-energy: ignoring unusable cache %s: %v\n", cachePath, err)
		case cached != nil:
			fmt.Fprintf(os.Stderr, "static self-energy: %v (loaded from %s — skipped the rebuild)\n",
				scheme, cachePath)
			return cached.Func(), nil
		}
	}

	sig, err := selfenergy.Static(ints, eps, nocc, norb, scheme, opts)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "static self-energy: %v\n", scheme)
	if cachePath != "off" {
		if err := writeSigmaCache(cachePath, key, sig); err != nil {
			// Not fatal — Σ is in memory and this run is fine. But say so loudly: the whole point
			// is that the NEXT generation does not spend days rebuilding it.
			fmt.Fprintf(os.Stderr, "static self-energy: WARNING could not cache to %s: %v "+
				"(the next run will rebuild Σ from scratch)\n", cachePath, err)
		} else {
			fmt.Fprintf(os.Stderr, "static self-energy: cached to %s\n", cachePath)
		}
	}
	return sig.Func(), nil
}
