; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @sme_block1(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) #0 {
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %14, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %15, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %16, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %17, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %19, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %18, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %20, 4, 1
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %9, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, ptr %10, 1
  %31 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %30, i64 %11, 2
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %31, i64 %12, 3, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, i64 %13, 4, 0
  %34 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %0, 0
  %35 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, ptr %1, 1
  %36 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %35, i64 %2, 2
  %37 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, i64 %3, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %37, i64 %6, 4, 0
  %39 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %38, i64 %4, 3, 1
  %40 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %39, i64 %7, 4, 1
  %41 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %40, i64 %5, 3, 2
  %42 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %41, i64 %8, 4, 2
  call void @llvm.aarch64.sme.zero(i32 17)
  %43 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 3, 0
  call void @llvm.aarch64.sme.zero(i32 34)
  call void @llvm.aarch64.sme.zero(i32 68)
  call void @llvm.aarch64.sme.zero(i32 136)
  br label %44

44:                                               ; preds = %175, %21
  %45 = phi i64 [ %176, %175 ], [ 0, %21 ]
  %46 = icmp slt i64 %45, %43
  br i1 %46, label %47, label %177

47:                                               ; preds = %50, %44
  %48 = phi i64 [ %174, %50 ], [ 0, %44 ]
  %49 = icmp slt i64 %48, 3
  br i1 %49, label %50, label %175

50:                                               ; preds = %47
  %51 = mul i64 %45, 9
  %52 = mul i64 %48, 3
  %53 = add i64 %51, %52
  %54 = mul i64 %53, 16
  %55 = add i64 %54, 16
  %56 = add i64 %54, 32
  %57 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %58 = getelementptr float, ptr %57, i64 %54
  %59 = load <vscale x 4 x float>, ptr %58, align 4
  %60 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %61 = getelementptr float, ptr %60, i64 %55
  %62 = load <vscale x 4 x float>, ptr %61, align 4
  %63 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, 1
  %64 = getelementptr float, ptr %63, i64 %56
  %65 = load <vscale x 4 x float>, ptr %64, align 4
  %66 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %67 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %68 = mul i64 %45, %67
  %69 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %70 = mul i64 %48, %69
  %71 = add i64 %68, %70
  %72 = add i64 %71, 0
  %73 = getelementptr float, ptr %66, i64 %72
  %74 = load <vscale x 4 x float>, ptr %73, align 4
  %75 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %76 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %77 = mul i64 %45, %76
  %78 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %79 = mul i64 %48, %78
  %80 = add i64 %77, %79
  %81 = add i64 %80, 1
  %82 = getelementptr float, ptr %75, i64 %81
  %83 = load <vscale x 4 x float>, ptr %82, align 4
  %84 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %85 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %86 = mul i64 %45, %85
  %87 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %88 = mul i64 %48, %87
  %89 = add i64 %86, %88
  %90 = add i64 %89, 2
  %91 = getelementptr float, ptr %84, i64 %90
  %92 = load <vscale x 4 x float>, ptr %91, align 4
  %93 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %94 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %95 = mul i64 %45, %94
  %96 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %97 = mul i64 %48, %96
  %98 = add i64 %95, %97
  %99 = add i64 %98, 16
  %100 = getelementptr float, ptr %93, i64 %99
  %101 = load <vscale x 4 x float>, ptr %100, align 4
  %102 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %103 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %104 = mul i64 %45, %103
  %105 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %106 = mul i64 %48, %105
  %107 = add i64 %104, %106
  %108 = add i64 %107, 17
  %109 = getelementptr float, ptr %102, i64 %108
  %110 = load <vscale x 4 x float>, ptr %109, align 4
  %111 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %112 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %113 = mul i64 %45, %112
  %114 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %115 = mul i64 %48, %114
  %116 = add i64 %113, %115
  %117 = add i64 %116, 18
  %118 = getelementptr float, ptr %111, i64 %117
  %119 = load <vscale x 4 x float>, ptr %118, align 4
  %120 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %121 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %122 = mul i64 %45, %121
  %123 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %124 = mul i64 %48, %123
  %125 = add i64 %122, %124
  %126 = add i64 %125, 32
  %127 = getelementptr float, ptr %120, i64 %126
  %128 = load <vscale x 4 x float>, ptr %127, align 4
  %129 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %130 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %131 = mul i64 %45, %130
  %132 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %133 = mul i64 %48, %132
  %134 = add i64 %131, %133
  %135 = add i64 %134, 33
  %136 = getelementptr float, ptr %129, i64 %135
  %137 = load <vscale x 4 x float>, ptr %136, align 4
  %138 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %139 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %140 = mul i64 %45, %139
  %141 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %142 = mul i64 %48, %141
  %143 = add i64 %140, %142
  %144 = add i64 %143, 34
  %145 = getelementptr float, ptr %138, i64 %144
  %146 = load <vscale x 4 x float>, ptr %145, align 4
  %147 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %148 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %149 = mul i64 %45, %148
  %150 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %151 = mul i64 %48, %150
  %152 = add i64 %149, %151
  %153 = add i64 %152, 48
  %154 = getelementptr float, ptr %147, i64 %153
  %155 = load <vscale x 4 x float>, ptr %154, align 4
  %156 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %157 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %158 = mul i64 %45, %157
  %159 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %160 = mul i64 %48, %159
  %161 = add i64 %158, %160
  %162 = add i64 %161, 49
  %163 = getelementptr float, ptr %156, i64 %162
  %164 = load <vscale x 4 x float>, ptr %163, align 4
  %165 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 1
  %166 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 0
  %167 = mul i64 %45, %166
  %168 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %42, 4, 1
  %169 = mul i64 %48, %168
  %170 = add i64 %167, %169
  %171 = add i64 %170, 50
  %172 = getelementptr float, ptr %165, i64 %171
  %173 = load <vscale x 4 x float>, ptr %172, align 4
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %59, <vscale x 4 x float> %74)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %62, <vscale x 4 x float> %83)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 1, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %65, <vscale x 4 x float> %92)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %59, <vscale x 4 x float> %101)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %62, <vscale x 4 x float> %110)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 2, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %65, <vscale x 4 x float> %119)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %59, <vscale x 4 x float> %128)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %62, <vscale x 4 x float> %137)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 3, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %65, <vscale x 4 x float> %146)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %59, <vscale x 4 x float> %155)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %62, <vscale x 4 x float> %164)
  call void @llvm.aarch64.sme.mopa.nxv4f32(i32 0, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x float> %65, <vscale x 4 x float> %173)
  %174 = add i64 %48, 1
  br label %47

175:                                              ; preds = %47
  %176 = add i64 %45, 1
  br label %44

177:                                              ; preds = %44
  %178 = call i64 @llvm.vscale.i64()
  %179 = mul i64 %178, 4
  br label %180

180:                                              ; preds = %183, %177
  %181 = phi i64 [ %190, %183 ], [ 0, %177 ]
  %182 = icmp slt i64 %181, %179
  br i1 %182, label %183, label %191

183:                                              ; preds = %180
  %184 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %185 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %186 = mul i64 %181, %185
  %187 = add i64 %186, 0
  %188 = getelementptr float, ptr %184, i64 %187
  %189 = trunc i64 %181 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %188, i32 1, i32 %189)
  %190 = add i64 %181, 1
  br label %180

191:                                              ; preds = %180
  br label %192

192:                                              ; preds = %195, %191
  %193 = phi i64 [ %202, %195 ], [ 0, %191 ]
  %194 = icmp slt i64 %193, %179
  br i1 %194, label %195, label %203

195:                                              ; preds = %192
  %196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %197 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %198 = mul i64 %193, %197
  %199 = add i64 %198, 16
  %200 = getelementptr float, ptr %196, i64 %199
  %201 = trunc i64 %193 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %200, i32 2, i32 %201)
  %202 = add i64 %193, 1
  br label %192

203:                                              ; preds = %192
  br label %204

204:                                              ; preds = %207, %203
  %205 = phi i64 [ %214, %207 ], [ 0, %203 ]
  %206 = icmp slt i64 %205, %179
  br i1 %206, label %207, label %215

207:                                              ; preds = %204
  %208 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %209 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %210 = mul i64 %205, %209
  %211 = add i64 %210, 32
  %212 = getelementptr float, ptr %208, i64 %211
  %213 = trunc i64 %205 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %212, i32 3, i32 %213)
  %214 = add i64 %205, 1
  br label %204

215:                                              ; preds = %204
  br label %216

216:                                              ; preds = %219, %215
  %217 = phi i64 [ %226, %219 ], [ 0, %215 ]
  %218 = icmp slt i64 %217, %179
  br i1 %218, label %219, label %227

219:                                              ; preds = %216
  %220 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %221 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %222 = mul i64 %217, %221
  %223 = add i64 %222, 48
  %224 = getelementptr float, ptr %220, i64 %223
  %225 = trunc i64 %217 to i32
  call void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer), ptr %224, i32 0, i32 %225)
  %226 = add i64 %217, 1
  br label %216

227:                                              ; preds = %216
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.zero(i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i64 @llvm.vscale.i64() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.st1w.horiz(<vscale x 4 x i1>, ptr, i32 immarg, i32) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.aarch64.sme.mopa.nxv4f32(i32 immarg, <vscale x 4 x i1>, <vscale x 4 x i1>, <vscale x 4 x float>, <vscale x 4 x float>) #1

attributes #0 = { "aarch64_new_za" "aarch64_pstate_sm_body" }
attributes #1 = { nocallback nofree nosync nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
