; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @sconv_sme_direct(i64 %0, i64 %1, i64 %2, i64 %3, i64 %4, i64 %5, ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, ptr %18, ptr %19, i64 %20, i64 %21, i64 %22, i64 %23, i64 %24) #0 {
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %18, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %19, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %20, 2
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 %21, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 %23, 4, 0
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 %22, 3, 1
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 %24, 4, 1
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %13, 0
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, ptr %14, 1
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 %15, 2
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 %16, 3, 0
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, i64 %17, 4, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %6, 0
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, ptr %7, 1
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 %8, 2
  %41 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, i64 %9, 3, 0
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %41, i64 %11, 4, 0
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, i64 %10, 3, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 %12, 4, 1
  %45 = call i64 @llvm.aarch64.sme.cntsw()
  %46 = sdiv i64 %3, %45
  br label %47

47:                                               ; preds = %245, %25
  %48 = phi i64 [ %246, %245 ], [ 0, %25 ]
  %49 = icmp slt i64 %48, %0
  br i1 %49, label %50, label %247

50:                                               ; preds = %47
  %51 = mul i64 %48, %1
  %52 = mul i64 %48, %3
  br label %53

53:                                               ; preds = %243, %50
  %54 = phi i64 [ %244, %243 ], [ 0, %50 ]
  %55 = icmp slt i64 %54, %46
  br i1 %55, label %56, label %245

56:                                               ; preds = %53
  %57 = mul i64 %54, %45
  %58 = add i64 %52, %57
  br label %59

59:                                               ; preds = %241, %56
  %60 = phi i64 [ %242, %241 ], [ 0, %56 ]
  %61 = icmp slt i64 %60, %4
  br i1 %61, label %62, label %243

62:                                               ; preds = %239, %59
  %63 = phi i64 [ %240, %239 ], [ 0, %59 ]
  %64 = icmp slt i64 %63, %5
  br i1 %64, label %65, label %241

65:                                               ; preds = %62
  call void @llvm.aarch64.sme.zero(i32 17)
  call void @llvm.aarch64.sme.zero(i32 34)
  call void @llvm.aarch64.sme.zero(i32 68)
  call void @llvm.aarch64.sme.zero(i32 136)
  br label %66

66:                                               ; preds = %178, %65
  %67 = phi i64 [ %179, %178 ], [ 0, %65 ]
  %68 = icmp slt i64 %67, %1
  br i1 %68, label %69, label %180

69:                                               ; preds = %66
  %70 = add i64 %51, %67
  %71 = mul i64 %67, 9
  br label %72

72:                                               ; preds = %75, %69
  %73 = phi i64 [ %177, %75 ], [ 0, %69 ]
  %74 = icmp slt i64 %73, 3
  br i1 %74, label %75, label %178

75:                                               ; preds = %72
  %76 = mul i64 %73, 3
  %77 = add i64 %71, %76
  %78 = mul i64 %77, %3
  %79 = add i64 %78, %57
  %80 = add i64 %79, %3
  %81 = add i64 %80, %3
  %82 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %83 = getelementptr float, ptr %82, i64 %79
  %84 = load <vscale x 4 x float>, ptr %83, align 4
  %85 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %86 = getelementptr float, ptr %85, i64 %80
  %87 = load <vscale x 4 x float>, ptr %86, align 4
  %88 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %89 = getelementptr float, ptr %88, i64 %81
  %90 = load <vscale x 4 x float>, ptr %89, align 4
  %91 = add i64 %60, %73
  %92 = mul i64 %91, %2
  %93 = add i64 %92, %63
  %94 = add i64 %93, %2
  %95 = add i64 %94, %2
  %96 = add i64 %95, %2
  %97 = add i64 %93, 1
  %98 = add i64 %93, 2
  %99 = add i64 %94, 1
  %100 = add i64 %94, 2
  %101 = add i64 %95, 1
  %102 = add i64 %95, 2
  %103 = add i64 %96, 1
  %104 = add i64 %96, 2
  %105 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %107 = mul i64 %70, %106
  %108 = add i64 %107, %93
  %109 = getelementptr float, ptr %105, i64 %108
  %110 = load <vscale x 4 x float>, ptr %109, align 4
  %111 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %112 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %113 = mul i64 %70, %112
  %114 = add i64 %113, %97
  %115 = getelementptr float, ptr %111, i64 %114
  %116 = load <vscale x 4 x float>, ptr %115, align 4
  %117 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %118 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %119 = mul i64 %70, %118
  %120 = add i64 %119, %98
  %121 = getelementptr float, ptr %117, i64 %120
  %122 = load <vscale x 4 x float>, ptr %121, align 4
  %123 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %124 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %125 = mul i64 %70, %124
  %126 = add i64 %125, %94
  %127 = getelementptr float, ptr %123, i64 %126
  %128 = load <vscale x 4 x float>, ptr %127, align 4
  %129 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %130 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %131 = mul i64 %70, %130
  %132 = add i64 %131, %99
  %133 = getelementptr float, ptr %129, i64 %132
  %134 = load <vscale x 4 x float>, ptr %133, align 4
  %135 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %136 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %137 = mul i64 %70, %136
  %138 = add i64 %137, %100
  %139 = getelementptr float, ptr %135, i64 %138
  %140 = load <vscale x 4 x float>, ptr %139, align 4
  %141 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %142 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %143 = mul i64 %70, %142
  %144 = add i64 %143, %95
  %145 = getelementptr float, ptr %141, i64 %144
  %146 = load <vscale x 4 x float>, ptr %145, align 4
  %147 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %148 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %149 = mul i64 %70, %148
  %150 = add i64 %149, %101
  %151 = getelementptr float, ptr %147, i64 %150
  %152 = load <vscale x 4 x float>, ptr %151, align 4
  %153 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %154 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %155 = mul i64 %70, %154
  %156 = add i64 %155, %102
  %157 = getelementptr float, ptr %153, i64 %156
  %158 = load <vscale x 4 x float>, ptr %157, align 4
  %159 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %160 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %161 = mul i64 %70, %160
  %162 = add i64 %161, %96
  %163 = getelementptr float, ptr %159, i64 %162
  %164 = load <vscale x 4 x float>, ptr %163, align 4
  %165 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %166 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %167 = mul i64 %70, %166
  %168 = add i64 %167, %103
  %169 = getelementptr float, ptr %165, i64 %168
  %170 = load <vscale x 4 x float>, ptr %169, align 4
  %171 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 1
  %172 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, 4, 0
  %173 = mul i64 %70, %172
  %174 = add i64 %173, %104
  %175 = getelementptr float, ptr %171, i64 %174
  %176 = load <vscale x 4 x float>, ptr %175, align 4
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %84, <vscale x 4 x float> %110)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %87, <vscale x 4 x float> %116)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %90, <vscale x 4 x float> %122)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %84, <vscale x 4 x float> %128)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %87, <vscale x 4 x float> %134)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %90, <vscale x 4 x float> %140)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %84, <vscale x 4 x float> %146)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %87, <vscale x 4 x float> %152)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %90, <vscale x 4 x float> %158)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %84, <vscale x 4 x float> %164)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %87, <vscale x 4 x float> %170)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %90, <vscale x 4 x float> %176)
  %177 = add i64 %73, 1
  br label %72

178:                                              ; preds = %72
  %179 = add i64 %67, 1
  br label %66

180:                                              ; preds = %66
  %181 = mul i64 %60, %5
  %182 = add i64 %181, %63
  %183 = add i64 %182, %5
  %184 = add i64 %183, %5
  %185 = add i64 %184, %5
  %186 = call i64 @llvm.vscale.i64()
  %187 = mul i64 %186, 4
  br label %188

188:                                              ; preds = %191, %180
  %189 = phi i64 [ %199, %191 ], [ 0, %180 ]
  %190 = icmp slt i64 %189, %187
  br i1 %190, label %191, label %200

191:                                              ; preds = %188
  %192 = add i64 %58, %189
  %193 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %194 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 0
  %195 = mul i64 %192, %194
  %196 = add i64 %195, %182
  %197 = getelementptr float, ptr %193, i64 %196
  %198 = trunc i64 %189 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %197, i32 1, i32 %198)
  %199 = add i64 %189, 1
  br label %188

200:                                              ; preds = %188
  br label %201

201:                                              ; preds = %204, %200
  %202 = phi i64 [ %212, %204 ], [ 0, %200 ]
  %203 = icmp slt i64 %202, %187
  br i1 %203, label %204, label %213

204:                                              ; preds = %201
  %205 = add i64 %58, %202
  %206 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %207 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 0
  %208 = mul i64 %205, %207
  %209 = add i64 %208, %183
  %210 = getelementptr float, ptr %206, i64 %209
  %211 = trunc i64 %202 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %210, i32 2, i32 %211)
  %212 = add i64 %202, 1
  br label %201

213:                                              ; preds = %201
  br label %214

214:                                              ; preds = %217, %213
  %215 = phi i64 [ %225, %217 ], [ 0, %213 ]
  %216 = icmp slt i64 %215, %187
  br i1 %216, label %217, label %226

217:                                              ; preds = %214
  %218 = add i64 %58, %215
  %219 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %220 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 0
  %221 = mul i64 %218, %220
  %222 = add i64 %221, %184
  %223 = getelementptr float, ptr %219, i64 %222
  %224 = trunc i64 %215 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %223, i32 3, i32 %224)
  %225 = add i64 %215, 1
  br label %214

226:                                              ; preds = %214
  br label %227

227:                                              ; preds = %230, %226
  %228 = phi i64 [ %238, %230 ], [ 0, %226 ]
  %229 = icmp slt i64 %228, %187
  br i1 %229, label %230, label %239

230:                                              ; preds = %227
  %231 = add i64 %58, %228
  %232 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %233 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 4, 0
  %234 = mul i64 %231, %233
  %235 = add i64 %234, %185
  %236 = getelementptr float, ptr %232, i64 %235
  %237 = trunc i64 %228 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %236, i32 0, i32 %237)
  %238 = add i64 %228, 1
  br label %227

239:                                              ; preds = %227
  %240 = add i64 %63, %45
  br label %62

241:                                              ; preds = %62
  %242 = add i64 %60, 4
  br label %59

243:                                              ; preds = %59
  %244 = add i64 %54, 1
  br label %53

245:                                              ; preds = %53
  %246 = add i64 %48, 1
  br label %47

247:                                              ; preds = %47
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
