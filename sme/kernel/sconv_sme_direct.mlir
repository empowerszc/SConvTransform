// Full SME direct convolution (stride1, 3x3, SAME pad), pre-padded I/O.
//
// Tiling: 4 ZA tiles are used for 4 consecutive OUTPUT ROWS (tile t -> row oh+t).
// Columns are tiled by the streaming vector length R (one tile = R columns).
// Padding is handled by the driver (spec §5.5): input and output are pre-padded
// so the kernel never reads/writes out of range and needs no masks.
//
// Parallelism: work is flattened into tasks over (n, ocb, 4-row block ohb).
//   task = ((n * Ocb) + ocb) * OhB + ohb,  OhB = OhP/4
// The caller runs tasks [T0, T0+TN) — one contiguous range per thread.
//
//   in:    memref<?x?xf32>  [N*Ic][IhP*IwP]  (pre-padded, row width IwP)
//   wpack: memref<?xf32>    [(Ic*9)*OcP]     index (ic*9+kh*3+kw)*OcP + oc
//   out:   memref<?x?xf32>  [N*OcP][OhP*OwP] (pre-padded, spatial = oh*OwP+ow)
//
// Scalars: N, Ic, IwP, OcP, OhP, OwP, T0, TN.
func.func @sconv_sme_compute(%N: index, %Ic: index, %IwP: index, %OcP: index,
                            %OhP: index, %OwP: index, %T0: index, %TN: index,
                            %in: memref<?x?xf32>, %wpack: memref<?xf32>,
                            %out: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c9 = arith.constant 9 : index

  %R = arm_sme.streaming_vl <word>
  %Ocb = arith.divsi %OcP, %R : index
  %OhB = arith.divsi %OhP, %c4 : index
  %T1 = arith.addi %T0, %TN : index

  scf.for %task = %T0 to %T1 step %c1 {
    %ohb = arith.remsi %task, %OhB : index
    %q = arith.divsi %task, %OhB : index
    %ocb = arith.remsi %q, %Ocb : index
    %n = arith.divsi %q, %Ocb : index
    %oh = arith.muli %ohb, %c4 : index
    %nIc = arith.muli %n, %Ic : index
    %nOcP = arith.muli %n, %OcP : index
    %row0 = arith.muli %ocb, %R : index
    %orow_out = arith.addi %nOcP, %row0 : index

    scf.for %c0i = %c0 to %OwP step %R {
      %za = arm_sme.zero : vector<[4]x[4]xf32>
      %zb = arm_sme.zero : vector<[4]x[4]xf32>
      %zc = arm_sme.zero : vector<[4]x[4]xf32>
      %zd = arm_sme.zero : vector<[4]x[4]xf32>
      %ra:4 = scf.for %ic = %c0 to %Ic step %c1
          iter_args(%ta = %za, %tb = %zb, %tc = %zc, %td = %zd)
          -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>) {
        %orow_in = arith.addi %nIc, %ic : index
        %ic9 = arith.muli %ic, %c9 : index
        %rb:4 = scf.for %kh = %c0 to %c3 step %c1
            iter_args(%a = %ta, %b = %tb, %c = %tc, %d = %td)
            -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>) {
          // weight base = (ic*9 + kh*3)*OcP + row0
          %kh3 = arith.muli %kh, %c3 : index
          %s = arith.addi %ic9, %kh3 : index
          %w0 = arith.muli %s, %OcP : index
          %wb = arith.addi %w0, %row0 : index
          %wb1 = arith.addi %wb, %OcP : index
          %wb2 = arith.addi %wb1, %OcP : index
          %wv0 = vector.load %wpack[%wb] : memref<?xf32>, vector<[4]xf32>
          %wv1 = vector.load %wpack[%wb1] : memref<?xf32>, vector<[4]xf32>
          %wv2 = vector.load %wpack[%wb2] : memref<?xf32>, vector<[4]xf32>

          // input padded row for tile t = oh + t + kh; flat col base = row*IwP + c0i
          %rr = arith.addi %oh, %kh : index
          %rrw = arith.muli %rr, %IwP : index
          %b0 = arith.addi %rrw, %c0i : index
          %b1 = arith.addi %b0, %IwP : index
          %b2 = arith.addi %b1, %IwP : index
          %b3 = arith.addi %b2, %IwP : index
          %b0a = arith.addi %b0, %c1 : index
          %b0b = arith.addi %b0, %c2 : index
          %b1a = arith.addi %b1, %c1 : index
          %b1b = arith.addi %b1, %c2 : index
          %b2a = arith.addi %b2, %c1 : index
          %b2b = arith.addi %b2, %c2 : index
          %b3a = arith.addi %b3, %c1 : index
          %b3b = arith.addi %b3, %c2 : index

          %i00 = vector.load %in[%orow_in, %b0] : memref<?x?xf32>, vector<[4]xf32>
          %i01 = vector.load %in[%orow_in, %b0a] : memref<?x?xf32>, vector<[4]xf32>
          %i02 = vector.load %in[%orow_in, %b0b] : memref<?x?xf32>, vector<[4]xf32>
          %i10 = vector.load %in[%orow_in, %b1] : memref<?x?xf32>, vector<[4]xf32>
          %i11 = vector.load %in[%orow_in, %b1a] : memref<?x?xf32>, vector<[4]xf32>
          %i12 = vector.load %in[%orow_in, %b1b] : memref<?x?xf32>, vector<[4]xf32>
          %i20 = vector.load %in[%orow_in, %b2] : memref<?x?xf32>, vector<[4]xf32>
          %i21 = vector.load %in[%orow_in, %b2a] : memref<?x?xf32>, vector<[4]xf32>
          %i22 = vector.load %in[%orow_in, %b2b] : memref<?x?xf32>, vector<[4]xf32>
          %i30 = vector.load %in[%orow_in, %b3] : memref<?x?xf32>, vector<[4]xf32>
          %i31 = vector.load %in[%orow_in, %b3a] : memref<?x?xf32>, vector<[4]xf32>
          %i32 = vector.load %in[%orow_in, %b3b] : memref<?x?xf32>, vector<[4]xf32>

          %a1 = arm_sme.outerproduct %wv0, %i00 acc(%a) : vector<[4]xf32>, vector<[4]xf32>
          %a2 = arm_sme.outerproduct %wv1, %i01 acc(%a1) : vector<[4]xf32>, vector<[4]xf32>
          %a3 = arm_sme.outerproduct %wv2, %i02 acc(%a2) : vector<[4]xf32>, vector<[4]xf32>
          %b1_ = arm_sme.outerproduct %wv0, %i10 acc(%b) : vector<[4]xf32>, vector<[4]xf32>
          %b2_ = arm_sme.outerproduct %wv1, %i11 acc(%b1_) : vector<[4]xf32>, vector<[4]xf32>
          %b3_ = arm_sme.outerproduct %wv2, %i12 acc(%b2_) : vector<[4]xf32>, vector<[4]xf32>
          %p1 = arm_sme.outerproduct %wv0, %i20 acc(%c) : vector<[4]xf32>, vector<[4]xf32>
          %p2 = arm_sme.outerproduct %wv1, %i21 acc(%p1) : vector<[4]xf32>, vector<[4]xf32>
          %p3 = arm_sme.outerproduct %wv2, %i22 acc(%p2) : vector<[4]xf32>, vector<[4]xf32>
          %q1 = arm_sme.outerproduct %wv0, %i30 acc(%d) : vector<[4]xf32>, vector<[4]xf32>
          %q2 = arm_sme.outerproduct %wv1, %i31 acc(%q1) : vector<[4]xf32>, vector<[4]xf32>
          %q3 = arm_sme.outerproduct %wv2, %i32 acc(%q2) : vector<[4]xf32>, vector<[4]xf32>

          scf.yield %a3, %b3_, %p3, %q3
            : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
        }
        scf.yield %rb#0, %rb#1, %rb#2, %rb#3
          : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
      }
      // store: tile t -> out row [n*OcP + row0], cols [(oh+t)*OwP + c0i]
      %s0 = arith.muli %oh, %OwP : index
      %s0c = arith.addi %s0, %c0i : index
      %s1 = arith.addi %s0c, %OwP : index
      %s2 = arith.addi %s1, %OwP : index
      %s3 = arith.addi %s2, %OwP : index
      arm_sme.tile_store %ra#0, %out[%orow_out, %s0c] : memref<?x?xf32>, vector<[4]x[4]xf32>
      arm_sme.tile_store %ra#1, %out[%orow_out, %s1] : memref<?x?xf32>, vector<[4]x[4]xf32>
      arm_sme.tile_store %ra#2, %out[%orow_out, %s2] : memref<?x?xf32>, vector<[4]x[4]xf32>
      arm_sme.tile_store %ra#3, %out[%orow_out, %s3] : memref<?x?xf32>, vector<[4]x[4]xf32>
    }
  }
  return
}
