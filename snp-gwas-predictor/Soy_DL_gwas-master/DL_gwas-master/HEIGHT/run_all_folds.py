import subprocess

NUM_FOLDS = 10

print(f"Updating data files with {NUM_FOLDS} folds.")

with open("fold_pcc_log.csv", "w") as f:
    f.write("")  # Truncate the file

for fold in range(1, NUM_FOLDS + 1):
    print(f"\n🚀 Running fold {fold}...\n")
    result = subprocess.run(['python3', 'height.py', '--fold', str(fold)])

    if result.returncode != 0:
        print(f"❌ Fold {fold} failed with return code {result.returncode}")
        break

# After all folds, run the full summary block
print("\n✅ All folds completed. Generating saliency summary and top SNPs...\n")
subprocess.run(['python3', 'height.py', '--summary'])