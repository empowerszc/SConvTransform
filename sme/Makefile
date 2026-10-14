LLVM ?= $(LLVM_BUILD_DIR)
BUILTINS := $(LLVM)/lib/clang/19/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a
CFLAGS := -O2
SME := -march=armv9-a+sme

all: smoke direct

smoke: kernel/sme_smoke.o test/smoke_main.o
	clang $(SME) $^ $(BUILTINS) -o smoke.out
	@./smoke.out

direct.out: kernel/sconv_sme_direct.o test/direct_main.o
	clang $(SME) $^ $(BUILTINS) -lpthread -o direct.out

direct: direct.out
	@./direct.out

bench: direct.out
	@./test/bench.sh

kernel/%.o: kernel/%.mlir build.sh
	./build.sh $<

test/%.o: test/%.c test/harness.h
	clang $(CFLAGS) -c $< -o $@

clean:
	rm -f kernel/*.ll kernel/*.o test/*.o smoke.out direct.out
.PHONY: all smoke direct bench clean
