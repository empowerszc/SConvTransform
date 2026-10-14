; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @sme_smoke(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16) #0 {
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %10, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, ptr %11, 1
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %12, 2
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %13, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 %15, 4, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 %14, 3, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %16, 4, 1
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %5, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, ptr %6, 1
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 %7, 2
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %8, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 %9, 4, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, ptr %1, 1
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, i64 %2, 2
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %3, 3, 0
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 %4, 4, 0
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 1
  %36 = getelementptr float, ptr %35, i64 0
  %37 = load <vscale x 4 x float>, ptr %36, align 4
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 1
  %39 = getelementptr float, ptr %38, i64 0
  %40 = load <vscale x 4 x float>, ptr %39, align 4
  call void @llvm.aarch64.sme.zero(i32 17)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %37, <vscale x 4 x float> %40)
  %41 = call i64 @llvm.vscale.i64()
  %42 = mul i64 %41, 4
  br label %43

43:                                               ; preds = %46, %17
  %44 = phi i64 [ %53, %46 ], [ 0, %17 ]
  %45 = icmp slt i64 %44, %42
  br i1 %45, label %46, label %54

46:                                               ; preds = %43
  %47 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 1
  %48 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 4, 0
  %49 = mul i64 %44, %48
  %50 = add i64 %49, 0
  %51 = getelementptr float, ptr %47, i64 %50
  %52 = trunc i64 %44 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %51, i32 0, i32 %52)
  %53 = add i64 %44, 1
  br label %43

54:                                               ; preds = %43
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.zero(i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.mopa.nxv4f32(i32 immarg, <vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i64 @llvm.vscale.i64() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1>, ptr, i32 immarg, i32) #1

attributes #0 = { "aarch64_new_za" "aarch64_pstate_sm_body" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
