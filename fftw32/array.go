package fftw32

// #include <fftw3.h>
import "C"

// Data for a 1D signal.
type Array struct {
	Elems []complex64
}

func (a *Array) Len() int {
	return len(a.Elems)
}

func (a *Array) At(i int) complex64 {
	return a.Elems[i]
}

func (a *Array) Set(i int, x complex64) {
	a.Elems[i] = x
}

func (a *Array) ptr() *complex64 {
	return &a.Elems[0]
}

// Allocates memory using fftw_malloc.
func NewArray(n int) *Array {
	elems := make([]complex64, n)
	return &Array{elems}
}

// 2D version of Array.
type Array2 struct {
	N     [2]int
	Elems []complex64
}

func (a *Array2) Dims() (int, int) {
	return a.N[0], a.N[1]
}

func (a *Array2) At(i0, i1 int) complex64 {
	return a.Elems[a.index(i0, i1)]
}

func (a *Array2) Set(i0, i1 int, x complex64) {
	a.Elems[a.index(i0, i1)] = x
}

func (a *Array2) index(i0, i1 int) int {
	return i1 + a.N[1]*i0
}

func (a *Array2) ptr() *complex64 {
	return &a.Elems[0]
}

func NewArray2(n0, n1 int) *Array2 {
	elems := make([]complex64, n0*n1)
	return &Array2{[...]int{n0, n1}, elems}
}

// 3D version of Array.
type Array3 struct {
	N     [3]int
	Elems []complex64
}

func (a *Array3) Dims() (int, int, int) {
	return a.N[0], a.N[1], a.N[2]
}

func (a *Array3) ptr() *complex64 {
	return &a.Elems[0]
}

func (a *Array3) At(i0, i1, i2 int) complex64 {
	return a.Elems[a.index(i0, i1, i2)]
}

func (a *Array3) Set(i0, i1, i2 int, x complex64) {
	a.Elems[a.index(i0, i1, i2)] = x
}

func (a *Array3) index(i0, i1, i2 int) int {
	return i2 + a.N[2]*(i1+i0*a.N[1])
}

func NewArray3(n0, n1, n2 int) *Array3 {
	elems := make([]complex64, n0*n1*n2)
	return &Array3{[...]int{n0, n1, n2}, elems}
}
