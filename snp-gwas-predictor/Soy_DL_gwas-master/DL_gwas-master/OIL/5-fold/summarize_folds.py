import os
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
from height import readData, get_saliency, plot_average_saliency, collect_saliency_across_folds, export_top_k_saliency, isru, indices_to_one_hot, nb_classes

IMP_input = "IMP_height.txt"
QA_input = "QA_height.txt"

NUM_FOLDS = 5

# Load data (again, just for saliency/summary)
imp_SNP, imp_pheno, folds, snp_names = readData(IMP_input)

# Collect saliency across models
print("\n📊 Collecting saliency across all folds...")
avg_saliency = collect_saliency_across_folds(imp_SNP, imp_pheno, folds)
plot_average_saliency(avg_saliency)

# After folds are run
if os.path.exists("fold_pcc_log.csv"):
	df = pd.read_csv("fold_pcc_log.csv", header=None, names=["Fold", "PCC_Imputed", "PCC_NonImputed"])
	df = df.sort_values("Fold")
	df.to_csv("fold_pcc_summary.csv", index=False)

	print("✅ Summary CSV generated: fold_pcc_summary.csv")
	print("📈 Average PCC (imputed):", round(df['PCC_Imputed'].mean(), 4))
	print("📈 Average PCC (non-imputed):", round(df['PCC_NonImputed'].mean(), 4))
else:
	print("❌ PCC log not found. Did folds run correctly?")
	
print("✅ Average saliency map and plot generated.")

# Visualize saliency for one sample
test_idx = np.where(folds == 1)[0]
sample = indices_to_one_hot(imp_SNP[test_idx[0]], nb_classes).astype(np.float32)

saliency_maps = []
for i in range(1, NUM_FOLDS + 1):
	model_path = f"model_IMP/model_{i}.h5"
	model = load_model(model_path, custom_objects={"isru": isru})
	model.compile(loss='mean_squared_error', optimizer='adam')

	sal = get_saliency(sample, model)
	saliency_maps.append(sal)

avg_saliency_map = np.mean(np.stack(saliency_maps), axis=0)
export_top_k_saliency(snp_names, avg_saliency_map, k=20)

# Interactive SNP viewer
while True:
	snp_query = input("Enter SNP name to view saliency (or type 'q' to quit): ")
	if snp_query.lower() in ['q', 'quit', 'exit']:
		break

	try:
		idx = snp_names.index(snp_query)
		print(f"🔬 Average saliency for {snp_query}: {avg_saliency_map[idx]:.6f}\n")
	except ValueError:
		print("❌ SNP not found. Please check the name and try again.\n")
		