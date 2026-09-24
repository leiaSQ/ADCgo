// Package sigma applies a generated ISR secular matrix as tensor contractions: the
// σ-build σ = M·Y of the isrgen packages without ever forming M.
//
// scripts/codegen/generate_adc.py --sigma derives, for every block and order, adcgen's
// matrix-vector product r_I = Σ_J M_IJ Y_J (SecularMatrix.mvp_block_order), integrates
// out the spin for each target spin block, and writes the terms as a Program: leaves
// (amplitude blocks, Coulomb integrals, intermediates, orbital-energy factors) and the
// pairwise contraction order that evaluates each term. This package runs a Program on
// a backend.
//
// The vector space is a khci.Space (one Ms sector of spin-orbital determinants in
// adcgen's precursor phase), so every generated package's per-element evaluator
// (Elem.BuildDense) is an exact reference for the operator applied here. Inside the
// apply, each class is held as spatial spin-block tensors: the spin-orbital amplitude
// restricted to one spin pattern, over the full spatial index ranges.
package sigma

// Kind is the kind of a term leaf.
type Kind uint8

const (
	// Amp is a block of the amplitude vector Y: class Class, spin pattern Spin.
	Amp Kind = iota
	// ERI is a Coulomb integral in chemist notation, (l0 l1|l2 l3) over spatial orbitals.
	ERI
	// Itmd is a spin block of a ground-state intermediate (t2_1, p0_2_oo, t2eri_*, ...),
	// built once from its own Program tensor.
	Itmd
	// Energy is a factor built from orbital energies (numerators and MP denominators),
	// a tensor over the labels its expression depends on.
	Energy
)

// Op is an instruction of an orbital-energy expression, evaluated as a stack machine.
type Op uint8

const (
	Const Op = iota // push Val
	Eps             // push the orbital energy of label Label
	Add             // pop N values, push their sum
	Mul             // pop N values, push their product
	Pow             // pop one value, push it to the integer power N (N != 0)
)

// Instr is one instruction of an Energy leaf's expression.
type Instr struct {
	Op    Op
	Label uint8
	N     int8
	Val   float64
}

// Leaf is an operand of a term. Labels name its modes; a label repeated within one leaf
// is a diagonal (both modes run together).
type Leaf struct {
	Kind   Kind
	Name   string  // Itmd: the intermediate
	Class  int     // Amp: excitation class (particles) of the amplitude block
	Spin   string  // Amp, Itmd: 'a' or 'b' per mode
	Labels []uint8 // one per mode
	Expr   []Instr // Energy: the expression; Labels are the labels it reads
}

// Step contracts operands A and B (indices into the term's leaves, then earlier steps'
// results in order) into a tensor over Out. Labels of A or B absent from Out and from
// the other operand are summed within that operand first.
type Step struct {
	A, B int
	Out  []uint8
}

// Term is one contribution Coef · (contraction of the leaves) to a target tensor. The
// result (the last step, or the single leaf) is accumulated into the target with its
// modes named by Target: a label repeated there is a Kronecker delta between target
// modes, and a target label the result does not carry is broadcast.
type Term struct {
	Block, Order int     // secular-matrix block (0..5 = B00 B01 B11 B02 B12 B22) and order; -1 for intermediates
	Coef         float64 //
	Spaces       string  // 'o' or 'v' per label
	Leaves       []Leaf
	Steps        []Step
	Target       []uint8
}

// Tensor is a target the program fills: a σ spin block (Name == ""), a diagonal block,
// or a spin block of an intermediate.
type Tensor struct {
	Name   string // intermediate name; "" for σ and diagonal blocks
	Class  int    // σ/diagonal: the class
	Spin   string // 'a' or 'b' per mode; σ blocks are alpha-first within particles and within holes
	Spaces string // 'o' or 'v' per mode
	Terms  []Term
}

// Program is a generated σ-build.
type Program struct {
	Variant string
	K       int      // main-class hole count
	Itmds   []Tensor // intermediates, dependencies first
	Sigma   []Tensor // σ spin blocks, every class and spin pattern the generator emitted
	Diag    []Tensor // diagonal spin blocks: diag(M) over the same patterns (no Amp leaves)
}
