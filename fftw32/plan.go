package fftw32

// #include <fftw3.h>
import "C"

import (
	"runtime"
	"sync"
	"unsafe"
)

// According to fftw's doc on multithreading, creation and destruction of plans should be single-
// threaded, so this will serve to synchronize that stuff, and hopefull multi-threaded is ok as long
// as it's all synchronous.
var createDestroyMu sync.Mutex

type Plan struct {
	fftwP C.fftwf_plan
	pin   runtime.Pinner
}

func (p *Plan) Execute() *Plan {
	C.fftwf_execute(p.fftwP)
	return p
}

func (p *Plan) Destroy() {
	createDestroyMu.Lock()
	if p.fftwP != nil {
		C.fftwf_destroy_plan(p.fftwP)
	}
	p.fftwP = nil
	createDestroyMu.Unlock()
	p.pin.Unpin()
}

func planFinalizer(p *Plan) {
	p.Destroy()
}

func NewPlan(in, out *Array, dir Direction, flag Flag) *Plan {
	// TODO: check that len(in) == len(out)
	plan := &Plan{}
	plan.pin.Pin(in.ptr())
	plan.pin.Pin(out.ptr())
	n := in.Len()
	var (
		n_    = C.int(n)
		in_   = (*C.fftwf_complex)(unsafe.Pointer(in.ptr()))
		out_  = (*C.fftwf_complex)(unsafe.Pointer(out.ptr()))
		dir_  = C.int(dir)
		flag_ = C.uint(flag)
	)
	createDestroyMu.Lock()
	plan.fftwP = C.fftwf_plan_dft_1d(n_, in_, out_, dir_, flag_)
	createDestroyMu.Unlock()
	runtime.SetFinalizer(plan, planFinalizer)
	return plan
}

func NewPlan2(in, out *Array2, dir Direction, flag Flag) *Plan {
	// TODO: check that in and out have the same dimensions
	plan := &Plan{}
	plan.pin.Pin(in.ptr())
	plan.pin.Pin(out.ptr())
	n0, n1 := in.Dims()
	var (
		n0_   = C.int(n0)
		n1_   = C.int(n1)
		in_   = (*C.fftwf_complex)(unsafe.Pointer(in.ptr()))
		out_  = (*C.fftwf_complex)(unsafe.Pointer(out.ptr()))
		dir_  = C.int(dir)
		flag_ = C.uint(flag)
	)
	createDestroyMu.Lock()
	plan.fftwP = C.fftwf_plan_dft_2d(n0_, n1_, in_, out_, dir_, flag_)
	createDestroyMu.Unlock()
	runtime.SetFinalizer(plan, planFinalizer)
	return plan
}

func NewPlan3(in, out *Array3, dir Direction, flag Flag) *Plan {
	// TODO: check that in and out have the same dimensions
	plan := &Plan{}
	plan.pin.Pin(in.ptr())
	plan.pin.Pin(out.ptr())
	n0, n1, n2 := in.Dims()
	var (
		n0_   = C.int(n0)
		n1_   = C.int(n1)
		n2_   = C.int(n2)
		in_   = (*C.fftwf_complex)(unsafe.Pointer(in.ptr()))
		out_  = (*C.fftwf_complex)(unsafe.Pointer(out.ptr()))
		dir_  = C.int(dir)
		flag_ = C.uint(flag)
	)
	createDestroyMu.Lock()
	plan.fftwP = C.fftwf_plan_dft_3d(n0_, n1_, n2_, in_, out_, dir_, flag_)
	createDestroyMu.Unlock()
	runtime.SetFinalizer(plan, planFinalizer)
	return plan
}
