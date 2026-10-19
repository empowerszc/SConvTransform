LLVM ?= $(LLVM_BUILD_DIR)
BUILTINS := $(LLVM)/lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a
CFLAGS := -O2
SME := -march=armv9-a+sme

all: smoke direct shim

smoke: kernel/sme_smoke.o test/smoke_main.o
	clang $(SME) $^ $(BUILTINS) -o smoke.out
	@./smoke.out

direct.out: kernel/sconv_sme_direct.o test/direct_main.o
	clang $(SME) $^ $(BUILTINS) -lpthread -o direct.out

direct: direct.out
	@./direct.out

shim: kernel/sconv_sme_direct.o runtime/sconv_sme_shim.o test/shim_main.o
	clang $(SME) $^ $(BUILTINS) -lpthread -o shim.out
	@./shim.out

e2e.out: kernel/sconv_sme_direct.o runtime/sconv_sme_shim.o test/shim_bench.o
	clang $(SME) $^ $(BUILTINS) -lpthread -o e2e.out

e2e: e2e.out
	@./e2e.out bench

runtime/%.o: runtime/%.c runtime/sme_abi.h
	clang $(CFLAGS) -c $< -o $@

bench: direct.out
	@./test/bench.sh

kernel/%.o: kernel/%.mlir build.sh
	./build.sh $<

test/%.o: test/%.c test/harness.h
	clang $(CFLAGS) -c $< -o $@

clean:
	rm -f kernel/*.ll kernel/*.o test/*.o runtime/*.o smoke.out direct.out shim.out e2e.out
.PHONY: all smoke direct bench shim e2e clean
