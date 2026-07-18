#!/usr/bin/env python3
"""Generate ConvBench MLIR payloads from a CSV dataset."""
import csv, sys, os

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <template.mlir> <dataset.csv> <output_dir> [--num N]")
        sys.exit(1)
    template_path, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    max_rows = None
    if "--num" in sys.argv:
        max_rows = int(sys.argv[sys.argv.index("--num") + 1])
    runs = 30
    if "--runs" in sys.argv:
        runs = int(sys.argv[sys.argv.index("--runs") + 1])
    os.makedirs(out_dir, exist_ok=True)
    with open(template_path) as f:
        template = f.read()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if max_rows and count >= max_rows:
                break
            phi = int(row["HI"]) + int(row["HPBOTTOM"]) + int(row["HPTOP"])
            pwi = int(row["WI"]) + int(row["WPLEFT"]) + int(row["WPRIGHT"])
            flops = 2 * int(row["HO"]) * int(row["WO"]) * int(row["DO"]) * int(row["CI"]) * int(row["HK"]) * int(row["WK"])
            rep = {"{{" + k + "}}": row[k] for k in row}
            rep["{{PHI}}"] = str(phi)
            rep["{{PWI}}"] = str(pwi)
            rep["{{FLOPS}}"] = str(flops)
            rep["{{RUNS}}"] = str(runs)
            mlir = template
            for k, v in rep.items():
                mlir = mlir.replace(k, v)
            out_file = os.path.join(out_dir, f"conv_{row['ID']}.mlir")
            with open(out_file, "w") as of:
                of.write(mlir)
            count += 1
            print(f"Generated: {out_file}")
    print(f"Total: {count} payloads")

if __name__ == "__main__":
    main()
