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

// A GPU backend may expose more than one physical device. counters maps its name to
// a device-count probe and multiCtors to a device-parameterized constructor; host
// backends register neither and are treated as a single device. These drive NewAll
// (one backend per visible device) and DeviceCount without changing the single-
// instance New/Register/Available surface every existing caller uses.
var (
	counters   = map[string]func() int{}
	multiCtors = map[string]func(dev int) Backend{}
)

// Register adds a backend constructor under name. Called from the init() of a
// build-tagged backend file.
func Register(name string, ctor func() Backend) { registry[name] = ctor }

// RegisterMulti registers a multi-device backend: count reports how many physical
// devices are visible and ctor builds a backend pinned to one of them. It also
// registers a plain single-instance entry (device 0) so New(name) and -backend name
// keep working unchanged. Called from a build-tagged GPU backend's init().
func RegisterMulti(name string, count func() int, ctor func(dev int) Backend) {
	Register(name, func() Backend { return ctor(0) })
	counters[name] = count
	multiCtors[name] = ctor
}

// DeviceCount reports how many devices the named backend can bind: the probe's value
// for a multi-device (GPU) backend, 1 for any other registered backend, 0 for an
// unknown name or a GPU backend with no visible device.
func DeviceCount(name string) int {
	if name == "" {
		name = "gonum"
	}
	if c, ok := counters[name]; ok {
		return c()
	}
	if _, ok := registry[name]; ok {
		return 1
	}
	return 0
}

// NewAll returns one backend instance per visible device for a multi-device backend,
// or a single instance for a host backend. maxDevices > 0 caps the count (0 = all
// visible); it never returns more than DeviceCount(name). The caller owns every
// returned backend and must Release them.
func NewAll(name string, maxDevices int) ([]Backend, error) {
	if name == "" {
		name = "gonum"
	}
	ctor, ok := multiCtors[name]
	if !ok {
		be, err := New(name) // host backend: a single instance
		if err != nil {
			return nil, err
		}
		return []Backend{be}, nil
	}
	n := counters[name]()
	if n <= 0 {
		return nil, fmt.Errorf("backend %q: no visible devices", name)
	}
	if maxDevices > 0 && maxDevices < n {
		n = maxDevices
	}
	out := make([]Backend, n)
	for i := range out {
		out[i] = ctor(i)
	}
	return out, nil
}

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
