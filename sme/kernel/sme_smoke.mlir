// out[i][j] = a[i] * b[j]  (16x16 outer product on the ZA tile)
func.func @sme_smoke(%a: memref<?xf32>, %b: memref<?xf32>, %out: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %va = vector.load %a[%c0] : memref<?xf32>, vector<[4]xf32>
  %vb = vector.load %b[%c0] : memref<?xf32>, vector<[4]xf32>
  %tile = arm_sme.outerproduct %va, %vb : vector<[4]xf32>, vector<[4]xf32>
  arm_sme.tile_store %tile, %out[%c0, %c0] : memref<?x?xf32>, vector<[4]x[4]xf32>
  return
}
