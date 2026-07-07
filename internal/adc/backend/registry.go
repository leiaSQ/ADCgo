package backend

import (
	"fmt"
	"sort"
)

// registry maps a backend name to its constructor. The pure-Go Gonum backend is
// always present; the accelerated backends register themselves from build-tagged
// files (openblas.go, hip.go, cuda.go) so a default build stays toolchain-free.
var registry = map[string]func() Backend{
	"gonum": func() Backend { return Gonum{} },
}

// Register adds a backend constructor under name. Called from the init() of a
// build-tagged backend file.
func Register(name string, ctor func() Backend) { registry[name] = ctor }

// New returns the named backend, or an error listing what this build offers. An
// empty name selects "gonum".
func New(name string) (Backend, error) {
	if name == "" {
		name = "gonum"
	}
	ctor, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("backend %q not available in this build (have: %v)", name, Available())
	}
	return ctor(), nil
}

// Available lists the backend names compiled into this build, sorted.
func Available() []string {
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
