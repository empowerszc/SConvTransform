// One output block: 16 oc (rows) x 64 output columns (4 ZA tiles), stride1, no pad.
//   in:     memref<?x?x?xf32>  [Ic][Ih][Iw]
//   wpack:  memref<?xf32>      [(Ic*9)][16]   (index: (ic*9+kh*3+kw)*16 + oc)
//   out:    memref<?x?xf32>    [16][64]
//
// For output row 0 (no pad) the input row for tap kh is simply kh.
// Tile t covers output columns [t*16, t*16+16); tap kw reads the input window
// starting at column t*16 + kw. Each tile accumulates 3 taps (kw=0,1,2).
func.func @sme_block1(%in: memref<?x?x?xf32>, %wpack: memref<?xf32>, %out: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c9 = arith.constant 9 : index
  %c16 = arith.constant 16 : index
  %c17 = arith.constant 17 : index
  %c18 = arith.constant 18 : index
  %c32 = arith.constant 32 : index
  %c33 = arith.constant 33 : index
  %c34 = arith.constant 34 : index
  %c48 = arith.constant 48 : index
  %c49 = arith.constant 49 : index
  %c50 = arith.constant 50 : index

  %ta = arm_sme.zero : vector<[4]x[4]xf32>
  %tb = arm_sme.zero : vector<[4]x[4]xf32>
  %tc = arm_sme.zero : vector<[4]x[4]xf32>
  %td = arm_sme.zero : vector<[4]x[4]xf32>

  %icn = memref.dim %in, %c0 : memref<?x?x?xf32>
  %res:4 = scf.for %ic = %c0 to %icn step %c1
      iter_args(%ra = %ta, %rb = %tb, %rc = %tc, %rd = %td)
      -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>) {
    %res2:4 = scf.for %kh = %c0 to %c3 step %c1
        iter_args(%xa = %ra, %xb = %rb, %xc = %rc, %xd = %rd)
        -> (vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>) {
      // weight base index = (ic*9 + kh*3) * 16
      %ic9 = arith.muli %ic, %c9 : index
      %kh3 = arith.muli %kh, %c3 : index
      %ws = arith.addi %ic9, %kh3 : index
      %w0 = arith.muli %ws, %c16 : index
      %w1 = arith.addi %w0, %c16 : index
      %w2 = arith.addi %w1, %c16 : index
      %wv0 = vector.load %wpack[%w0] : memref<?xf32>, vector<[4]xf32>
      %wv1 = vector.load %wpack[%w1] : memref<?xf32>, vector<[4]xf32>
      %wv2 = vector.load %wpack[%w2] : memref<?xf32>, vector<[4]xf32>

      // tile 0: input columns 0,1,2
      %i00 = vector.load %in[%ic, %kh, %c0] : memref<?x?x?xf32>, vector<[4]xf32>
      %i01 = vector.load %in[%ic, %kh, %c1] : memref<?x?x?xf32>, vector<[4]xf32>
      %i02 = vector.load %in[%ic, %kh, %c2] : memref<?x?x?xf32>, vector<[4]xf32>
      // tile 1: input columns 16,17,18
      %i10 = vector.load %in[%ic, %kh, %c16] : memref<?x?x?xf32>, vector<[4]xf32>
      %i11 = vector.load %in[%ic, %kh, %c17] : memref<?x?x?xf32>, vector<[4]xf32>
      %i12 = vector.load %in[%ic, %kh, %c18] : memref<?x?x?xf32>, vector<[4]xf32>
      // tile 2: input columns 32,33,34
      %i20 = vector.load %in[%ic, %kh, %c32] : memref<?x?x?xf32>, vector<[4]xf32>
      %i21 = vector.load %in[%ic, %kh, %c33] : memref<?x?x?xf32>, vector<[4]xf32>
      %i22 = vector.load %in[%ic, %kh, %c34] : memref<?x?x?xf32>, vector<[4]xf32>
      // tile 3: input columns 48,49,50
      %i30 = vector.load %in[%ic, %kh, %c48] : memref<?x?x?xf32>, vector<[4]xf32>
      %i31 = vector.load %in[%ic, %kh, %c49] : memref<?x?x?xf32>, vector<[4]xf32>
      %i32 = vector.load %in[%ic, %kh, %c50] : memref<?x?x?xf32>, vector<[4]xf32>

      %a1 = arm_sme.outerproduct %wv0, %i00 acc(%xa) : vector<[4]xf32>, vector<[4]xf32>
      %a2 = arm_sme.outerproduct %wv1, %i01 acc(%a1) : vector<[4]xf32>, vector<[4]xf32>
      %a3 = arm_sme.outerproduct %wv2, %i02 acc(%a2) : vector<[4]xf32>, vector<[4]xf32>
      %b1 = arm_sme.outerproduct %wv0, %i10 acc(%xb) : vector<[4]xf32>, vector<[4]xf32>
      %b2 = arm_sme.outerproduct %wv1, %i11 acc(%b1) : vector<[4]xf32>, vector<[4]xf32>
      %b3 = arm_sme.outerproduct %wv2, %i12 acc(%b2) : vector<[4]xf32>, vector<[4]xf32>
      %p1 = arm_sme.outerproduct %wv0, %i20 acc(%xc) : vector<[4]xf32>, vector<[4]xf32>
      %p2 = arm_sme.outerproduct %wv1, %i21 acc(%p1) : vector<[4]xf32>, vector<[4]xf32>
      %p3 = arm_sme.outerproduct %wv2, %i22 acc(%p2) : vector<[4]xf32>, vector<[4]xf32>
      %q1 = arm_sme.outerproduct %wv0, %i30 acc(%xd) : vector<[4]xf32>, vector<[4]xf32>
      %q2 = arm_sme.outerproduct %wv1, %i31 acc(%q1) : vector<[4]xf32>, vector<[4]xf32>
      %q3 = arm_sme.outerproduct %wv2, %i32 acc(%q2) : vector<[4]xf32>, vector<[4]xf32>

      scf.yield %a3, %b3, %p3, %q3
        : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
    }
    scf.yield %res2#0, %res2#1, %res2#2, %res2#3
      : vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>, vector<[4]x[4]xf32>
  }

  %c0b = arith.constant 0 : index
  arm_sme.tile_store %res#0, %out[%c0b, %c0b] : memref<?x?xf32>, vector<[4]x[4]xf32>
  arm_sme.tile_store %res#1, %out[%c0b, %c16] : memref<?x?xf32>, vector<[4]x[4]xf32>
  arm_sme.tile_store %res#2, %out[%c0b, %c32] : memref<?x?xf32>, vector<[4]x[4]xf32>
  arm_sme.tile_store %res#3, %out[%c0b, %c48] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}
