; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @sconv_sme_compute(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, ptr %8, ptr %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, ptr %15, ptr %16, i64 %17, i64 %18, i64 %19, ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 %25, i64 %26) #0 {
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %20, 0
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, ptr %21, 1
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %22, 2
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %23, 3, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %25, 4, 0
  %33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, i64 %24, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %33, i64 %26, 4, 1
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %15, 0
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, ptr %16, 1
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, i64 %17, 2
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, i64 %18, 3, 0
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 %19, 4, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %8, 0
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, ptr %9, 1
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %10, 2
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %11, 3, 0
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %13, 4, 0
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 %12, 3, 1
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 %14, 4, 1
  %47 = call i64 @llvm.aarch64.sme.cntsw()
  %48 = sdiv i64 %3, %47
  %49 = sdiv i64 %4, 4
  %50 = add i64 %6, %7
  br label %51

51:                                               ; preds = %243, %27
  %52 = phi i64 [ %244, %243 ], [ %6, %27 ]
  %53 = icmp slt i64 %52, %50
  br i1 %53, label %54, label %245

54:                                               ; preds = %51
  %55 = srem i64 %52, %49
  %56 = sdiv i64 %52, %49
  %57 = srem i64 %56, %48
  %58 = sdiv i64 %56, %48
  %59 = mul i64 %55, 4
  %60 = mul i64 %58, %1
  %61 = mul i64 %58, %3
  %62 = mul i64 %57, %47
  %63 = add i64 %61, %62
  br label %64

64:                                               ; preds = %241, %54
  %65 = phi i64 [ %242, %241 ], [ 0, %54 ]
  %66 = icmp slt i64 %65, %5
  br i1 %66, label %67, label %243

67:                                               ; preds = %64
  call void @llvm.aarch64.sme.zero(i32 17)
  call void @llvm.aarch64.sme.zero(i32 34)
  call void @llvm.aarch64.sme.zero(i32 68)
  call void @llvm.aarch64.sme.zero(i32 136)
  br label %68

68:                                               ; preds = %180, %67
  %69 = phi i64 [ %181, %180 ], [ 0, %67 ]
  %70 = icmp slt i64 %69, %1
  br i1 %70, label %71, label %182

71:                                               ; preds = %68
  %72 = add i64 %60, %69
  %73 = mul i64 %69, 9
  br label %74

74:                                               ; preds = %77, %71
  %75 = phi i64 [ %179, %77 ], [ 0, %71 ]
  %76 = icmp slt i64 %75, 3
  br i1 %76, label %77, label %180

77:                                               ; preds = %74
  %78 = mul i64 %75, 3
  %79 = add i64 %73, %78
  %80 = mul i64 %79, %3
  %81 = add i64 %80, %62
  %82 = add i64 %81, %3
  %83 = add i64 %82, %3
  %84 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, 1
  %85 = getelementptr float, ptr %84, i64 %81
  %86 = load <vscale x 4 x float>, ptr %85, align 4
  %87 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, 1
  %88 = getelementptr float, ptr %87, i64 %82
  %89 = load <vscale x 4 x float>, ptr %88, align 4
  %90 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, 1
  %91 = getelementptr float, ptr %90, i64 %83
  %92 = load <vscale x 4 x float>, ptr %91, align 4
  %93 = add i64 %59, %75
  %94 = mul i64 %93, %2
  %95 = add i64 %94, %65
  %96 = add i64 %95, %2
  %97 = add i64 %96, %2
  %98 = add i64 %97, %2
  %99 = add i64 %95, 1
  %100 = add i64 %95, 2
  %101 = add i64 %96, 1
  %102 = add i64 %96, 2
  %103 = add i64 %97, 1
  %104 = add i64 %97, 2
  %105 = add i64 %98, 1
  %106 = add i64 %98, 2
  %107 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %108 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %109 = mul i64 %72, %108
  %110 = add i64 %109, %95
  %111 = getelementptr float, ptr %107, i64 %110
  %112 = load <vscale x 4 x float>, ptr %111, align 4
  %113 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %114 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %115 = mul i64 %72, %114
  %116 = add i64 %115, %99
  %117 = getelementptr float, ptr %113, i64 %116
  %118 = load <vscale x 4 x float>, ptr %117, align 4
  %119 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %120 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %121 = mul i64 %72, %120
  %122 = add i64 %121, %100
  %123 = getelementptr float, ptr %119, i64 %122
  %124 = load <vscale x 4 x float>, ptr %123, align 4
  %125 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %126 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %127 = mul i64 %72, %126
  %128 = add i64 %127, %96
  %129 = getelementptr float, ptr %125, i64 %128
  %130 = load <vscale x 4 x float>, ptr %129, align 4
  %131 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %132 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %133 = mul i64 %72, %132
  %134 = add i64 %133, %101
  %135 = getelementptr float, ptr %131, i64 %134
  %136 = load <vscale x 4 x float>, ptr %135, align 4
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %139 = mul i64 %72, %138
  %140 = add i64 %139, %102
  %141 = getelementptr float, ptr %137, i64 %140
  %142 = load <vscale x 4 x float>, ptr %141, align 4
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %144 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %145 = mul i64 %72, %144
  %146 = add i64 %145, %97
  %147 = getelementptr float, ptr %143, i64 %146
  %148 = load <vscale x 4 x float>, ptr %147, align 4
  %149 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %150 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %151 = mul i64 %72, %150
  %152 = add i64 %151, %103
  %153 = getelementptr float, ptr %149, i64 %152
  %154 = load <vscale x 4 x float>, ptr %153, align 4
  %155 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %156 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %157 = mul i64 %72, %156
  %158 = add i64 %157, %104
  %159 = getelementptr float, ptr %155, i64 %158
  %160 = load <vscale x 4 x float>, ptr %159, align 4
  %161 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %162 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %163 = mul i64 %72, %162
  %164 = add i64 %163, %98
  %165 = getelementptr float, ptr %161, i64 %164
  %166 = load <vscale x 4 x float>, ptr %165, align 4
  %167 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %168 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %169 = mul i64 %72, %168
  %170 = add i64 %169, %105
  %171 = getelementptr float, ptr %167, i64 %170
  %172 = load <vscale x 4 x float>, ptr %171, align 4
  %173 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 1
  %174 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, 4, 0
  %175 = mul i64 %72, %174
  %176 = add i64 %175, %106
  %177 = getelementptr float, ptr %173, i64 %176
  %178 = load <vscale x 4 x float>, ptr %177, align 4
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %86, <vscale x 4 x float> %112)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %89, <vscale x 4 x float> %118)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %92, <vscale x 4 x float> %124)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %86, <vscale x 4 x float> %130)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %89, <vscale x 4 x float> %136)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %92, <vscale x 4 x float> %142)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %86, <vscale x 4 x float> %148)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %89, <vscale x 4 x float> %154)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %92, <vscale x 4 x float> %160)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %86, <vscale x 4 x float> %166)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %89, <vscale x 4 x float> %172)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %92, <vscale x 4 x float> %178)
  %179 = add i64 %75, 1
  br label %74

180:                                              ; preds = %74
  %181 = add i64 %69, 1
  br label %68

182:                                              ; preds = %68
  %183 = mul i64 %59, %5
  %184 = add i64 %183, %65
  %185 = add i64 %184, %5
  %186 = add i64 %185, %5
  %187 = add i64 %186, %5
  %188 = call i64 @llvm.vscale.i64()
  %189 = mul i64 %188, 4
  br label %190

190:                                              ; preds = %193, %182
  %191 = phi i64 [ %201, %193 ], [ 0, %182 ]
  %192 = icmp slt i64 %191, %189
  br i1 %192, label %193, label %202

193:                                              ; preds = %190
  %194 = add i64 %63, %191
  %195 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 4, 0
  %197 = mul i64 %194, %196
  %198 = add i64 %197, %184
  %199 = getelementptr float, ptr %195, i64 %198
  %200 = trunc i64 %191 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %199, i32 1, i32 %200)
  %201 = add i64 %191, 1
  br label %190

202:                                              ; preds = %190
  br label %203

203:                                              ; preds = %206, %202
  %204 = phi i64 [ %214, %206 ], [ 0, %202 ]
  %205 = icmp slt i64 %204, %189
  br i1 %205, label %206, label %215

206:                                              ; preds = %203
  %207 = add i64 %63, %204
  %208 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %209 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 4, 0
  %210 = mul i64 %207, %209
  %211 = add i64 %210, %185
  %212 = getelementptr float, ptr %208, i64 %211
  %213 = trunc i64 %204 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %212, i32 2, i32 %213)
  %214 = add i64 %204, 1
  br label %203

215:                                              ; preds = %203
  br label %216

216:                                              ; preds = %219, %215
  %217 = phi i64 [ %227, %219 ], [ 0, %215 ]
  %218 = icmp slt i64 %217, %189
  br i1 %218, label %219, label %228

219:                                              ; preds = %216
  %220 = add i64 %63, %217
  %221 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %222 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 4, 0
  %223 = mul i64 %220, %222
  %224 = add i64 %223, %186
  %225 = getelementptr float, ptr %221, i64 %224
  %226 = trunc i64 %217 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %225, i32 3, i32 %226)
  %227 = add i64 %217, 1
  br label %216

228:                                              ; preds = %216
  br label %229

229:                                              ; preds = %232, %228
  %230 = phi i64 [ %240, %232 ], [ 0, %228 ]
  %231 = icmp slt i64 %230, %189
  br i1 %231, label %232, label %241

232:                                              ; preds = %229
  %233 = add i64 %63, %230
  %234 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 1
  %235 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, 4, 0
  %236 = mul i64 %233, %235
  %237 = add i64 %236, %187
  %238 = getelementptr float, ptr %234, i64 %237
  %239 = trunc i64 %230 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %238, i32 0, i32 %239)
  %240 = add i64 %230, 1
  br label %229

241:                                              ; preds = %229
  %242 = add i64 %65, %47
  br label %64

243:                                              ; preds = %64
  %244 = add i64 %52, 1
  br label %51

245:                                              ; preds = %51
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i64 @llvm.aarch64.sme.cntsw() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.zero(i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i64 @llvm.vscale.i64() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1>, ptr, i32 immarg, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.mopa.nxv4f32(i32 immarg, <vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>) #2

attributes #0 = { "aarch64_new_za" "aarch64_pstate_sm_body" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
