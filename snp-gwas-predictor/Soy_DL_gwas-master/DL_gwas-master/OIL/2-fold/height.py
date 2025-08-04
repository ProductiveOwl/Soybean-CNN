from __future__ import print_function
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
import numpy as np
import random
import pandas as pd
from scipy import stats
import sys
import logging
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import *
from keras.regularizers import l1,l2, L1L2
from keras import backend as K
import gc
from sklearn.metrics.pairwise import cosine_similarity
import keras
import keras.utils as kutils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint,ReduceLROnPlateau
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets, linear_model
import itertools
from keras.models import load_model
import csv
import argparse
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math as m
import keras.backend as K
import sklearn 

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2" #New
tf.config.set_visible_devices([], 'GPU') #New
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

##one of K encoding


nb_classes = 4

NUM_FOLDS = 2

def assign_folds_to_file(filename, output_filename=None, seed=42):
	if output_filename is None:
		output_filename = filename  # Overwrite original
	
	df = pd.read_csv(filename, sep='\t')
	num_samples = len(df)

	# Generate new fold assignments
	np.random.seed(seed)
	new_folds = np.tile(np.arange(1, NUM_FOLDS + 1), int(np.ceil(num_samples / NUM_FOLDS)))[:num_samples]
	assert len(new_folds) == num_samples, "Mismatch in fold assignment length!"

	df.iloc[:, 0] = new_folds  # Replace first column with new folds

	# Double check before writing
	assert len(df) == num_samples, f"Row count changed! Expected {num_samples}, got {len(df)}"

	df.to_csv(output_filename, sep='\t', index=False)

def sync_folds_column(source_file, target_file, output_file=None):
	"""
	Copy the first column (fold assignment) from source_file to target_file.
	Optionally write the result to output_file (or overwrite target_file).
	"""
	if output_file is None:
		output_file = target_file

	source_df = pd.read_csv(source_file, sep='\t')
	target_df = pd.read_csv(target_file, sep='\t')

	if len(source_df) != len(target_df):
		raise ValueError(f"Row count mismatch: {source_file} has {len(source_df)}, {target_file} has {len(target_df)}")

	target_df.iloc[:, 0] = source_df.iloc[:, 0]  # Copy fold column
	target_df.to_csv(output_file, sep='\t', index=False)

def indices_to_one_hot(data,nb_classes):
	
	targets = np.array(data).reshape(-1)
	
	return np.eye(nb_classes)[targets]

def predict_height_from_all_folds(snp_vector, model_dir="model_IMP"):
	one_hot = indices_to_one_hot(snp_vector, nb_classes)
	one_hot = np.expand_dims(one_hot, axis=0).astype(np.float32)

	predictions = []

	for i in range(1, NUM_FOLDS + 1):
		model_path = f"{model_dir}/model_{i}.h5"
		model = load_model(model_path, custom_objects={"isru": isru})
		model.compile(loss='mean_squared_error', optimizer='adam')

		pred = model.predict(one_hot, verbose=0)
		predictions.append(pred[0][0])

	return np.mean(predictions)	

def readData(input):
	
	data = pd.read_csv(input,sep='\t',header=0,na_values='nan')
	snp_df = data.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
	SNP = snp_df.values
	snp_names = snp_df.columns.tolist()
	pheno = data.iloc[:,1].apply(pd.to_numeric, errors='coerce').values
	folds = data.iloc[:,0].apply(pd.to_numeric, errors='coerce').values
	
	#arr = np.empty(shape=(SNP.shape[0],SNP.shape[1] , nb_classes))
	# arr = np.memmap('snp_encoded.dat', dtype='float32', mode='w+', shape=(SNP.shape[0], SNP.shape[1], nb_classes))
	
	# for i in range(0,SNP.shape[0]):
	# 	arr[i] = indices_to_one_hot(pd.to_numeric(SNP[i],downcast='signed'), nb_classes)
		
	return SNP.astype(np.int8), pheno, folds, snp_names
	
def resnet(input):
	
	inputs = Input(shape=(input.shape[1],nb_classes))
	
	
	x = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(inputs)
	
	x = Conv1D(10,20,padding='same',activation = 'linear', kernel_initializer = 'TruncatedNormal',kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(x)
		
	x = Dropout(0.75)(x)
	
	shortcut = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(inputs)
	x = layers.add([shortcut,x])
	
	x = Conv1D(10,4,padding='same',activation = 'linear',kernel_initializer = 'TruncatedNormal', kernel_regularizer=regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.01))(x)
	
	x = Dropout(0.75)(x)
	x = Flatten()(x)
	
	x = Dropout(0.75)(x)
	
	outputs = Dense(1,activation = isru,bias_regularizer = regularizers.l2(0.01),kernel_initializer = 'TruncatedNormal',name = 'out')(x)
	
	model = Model(inputs = inputs,outputs = outputs)
	model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])
	
	return model	
	
def show_images_plot(saliency,wald,outname):
	
	plt.figure(figsize=(15, 8), facecolor='w')
	
	plt.subplot(2, 1, 1)
	x = np.median(saliency,axis=-1)
	plt.plot(x,'b.')
	line = sorted(x,reverse = True)[10]
	plt.axhline(y = line,color='b', linestyle='--')
	plt.ylabel('saliency value', fontdict=None, labelpad=None,fontsize=15)
	
	
	plt.subplot(2, 1, 2)
	plt.plot(wald,'r1')
	line = sorted(wald,reverse = True)[10]
	plt.axhline(y = line,color='r', linestyle='--')
	
	plt.xlabel('SNPs', fontdict=None, labelpad=None,fontsize=15)
	plt.ylabel('Wald', fontdict=None, labelpad=None,fontsize=15)
	
	plt.savefig(outname)
	plt.clf()
	plt.cla()
	plt.close()

def plot_average_saliency(avg_saliency, output_file="avg_saliency_across_folds.png"):
	plt.figure(figsize=(15, 6))
	plt.plot(avg_saliency, '.', markersize=3)
	plt.xlabel("SNP Index")
	plt.ylabel("Average Saliency")
	plt.title("Average Saliency per SNP Across All Folds")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()	
	
def get_saliency(input_tensor, model):
	"""
	Compute saliency map using TensorFlow GradientTape.
	Args:
		input_tensor: one-hot encoded SNPs for a single sample (shape: [num_SNPs, 4])
		model: trained Keras model
	Returns:
		saliency: np.array of shape (num_SNPs,) with max gradient across channels
	"""
	input_tensor = tf.convert_to_tensor(np.expand_dims(input_tensor, axis=0))  # shape: (1, num_SNPs, 4)
	input_tensor = tf.cast(input_tensor, tf.float32)

	with tf.GradientTape() as tape:
		tape.watch(input_tensor)
		prediction = model(input_tensor, training=False)  # shape: (1, 1)

	gradient = tape.gradient(prediction, input_tensor)  # shape: (1, num_SNPs, 4)
	saliency = tf.reduce_max(tf.abs(gradient), axis=-1)  # max across classes
	return saliency.numpy().squeeze()  # shape: (num_SNPs,)

def export_top_k_saliency(snp_names, saliency_values, k=20, output_file="top_saliency_snps.csv"):
	# Sort SNPs by saliency (descending)
	top_indices = np.argsort(saliency_values)[::-1][:k]
	top_snps = [(snp_names[i], saliency_values[i]) for i in top_indices]

	# Write to CSV
	with open(output_file, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["SNP", "Saliency"])
		for snp, score in top_snps:
			writer.writerow([snp, f"{score:.6f}"])
	print(f"📁 Top {k} SNPs by saliency saved to '{output_file}'")

def collect_saliency_across_folds(imp_SNP, imp_pheno, folds):
	all_saliencies = []

	for i in range(1, NUM_FOLDS + 1):
		print(f"Processing fold {i}...")

		# Load the trained model for this fold
		model_path = f"model_IMP/model_{i}.h5"
		model = load_model(model_path, custom_objects={"isru": isru})
		model.compile(loss='mean_squared_error', optimizer='adam')

		# Get test indices for this fold
		test_idx = np.where(folds == i)[0]

		# Compute saliency for each test sample
		for idx in test_idx:
			snp_vector = indices_to_one_hot(imp_SNP[idx], nb_classes).astype(np.float32)
			sal = get_saliency(snp_vector, model)
			all_saliencies.append(sal)

	print("Finished collecting saliency data from all folds.")
	return np.mean(np.stack(all_saliencies), axis=0)  # shape: (num_SNPs,)

a= 0.03  #height

def isru(x):
	return  x / (tf.math.sqrt(1 + a * tf.math.square(x)))

class SNPGenerator(tf.keras.utils.Sequence):
	def __init__(self, SNP, labels, batch_size, **kwargs):
		super().__init__(**kwargs)
		self.SNP = SNP
		self.labels = labels
		self.batch_size = batch_size

	def __len__(self):
		return int(np.ceil(len(self.SNP) / self.batch_size))

	def __getitem__(self, idx):
		batch_x = self.SNP[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_encoded = np.array([indices_to_one_hot(x, nb_classes) for x in batch_x], dtype=np.float32)
		return batch_encoded, batch_y

def model_train(testSNP, valSNP, trainSNP, testPheno, valPheno, trainPheno, model_save, weights_save):
	batch_size = 4
	early_stopping = keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, mode='min')

	model = resnet(trainSNP)

	# Use generators for training and validation
	train_gen = SNPGenerator(trainSNP, trainPheno, batch_size)
	val_gen = SNPGenerator(valSNP, valPheno, batch_size)

	history = model.fit(
		train_gen,
		epochs=1000,
		validation_data=val_gen,
		callbacks=[early_stopping],
		shuffle=True,
		verbose=1
	)

	model.save(model_save)
	model.save_weights(weights_save)

	# Test set: one-hot encode manually
	test_encoded = np.array([indices_to_one_hot(x, nb_classes) for x in testSNP], dtype=np.float32)
	pred = model.predict(test_encoded)
	pred = pred.flatten()

	corr = pearsonr(pred, testPheno)[0]
	return history, corr
	
def safe_delete(*varnames):
	for name in varnames:
		if name in locals():
			del locals()[name]
		elif name in globals():
			del globals()[name]

def run_saliency_summary(IMP_input, QA_input):
	imp_SNP, imp_pheno, folds, snp_names = readData(IMP_input)

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

	# SNP saliency viewer
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

def main(IMP_input, QA_input, run_fold=None):
	IMP_corr = []
	QA_corr = []

	# Load data once
	imp_SNP, imp_pheno, folds, snp_names = readData(IMP_input)
	QA_SNP, QA_pheno, folds, _ = readData(QA_input)
	PHENOTYPE = imp_pheno

	# If a specific fold is requested, just run that one; otherwise run all
	fold_range = [run_fold] if run_fold else range(1, NUM_FOLDS + 1)

	for i in fold_range:
		print(f"\n🔁 Starting fold {i}...")

		# Identify test fold
		testIdx = np.where(folds == i)[0]

		# If NUM_FOLDS >= 3, use separate fold for validation
		if NUM_FOLDS >= 3:
			val_fold = (i % NUM_FOLDS) + 1
			valIdx = np.where(folds == val_fold)[0]
			trainIdx = np.where((folds != i) & (folds != val_fold))[0]
		else:
			# With 2 folds: split the other fold randomly into train/val
			other_idx = np.where(folds != i)[0]
			np.random.seed(42)
			np.random.shuffle(other_idx)
			val_size = int(0.2 * len(other_idx))
			valIdx = other_idx[:val_size]
			trainIdx = other_idx[val_size:]

		if len(trainIdx) == 0 or len(valIdx) == 0:
			raise ValueError(f"Fold {i}: Training or validation set is empty. Check NUM_FOLDS or data distribution.")

		# Partition data
		trainSNP, trainSNP_QA, trainPheno = imp_SNP[trainIdx], QA_SNP[trainIdx], PHENOTYPE[trainIdx]
		valSNP, valSNP_QA, valPheno = imp_SNP[valIdx], QA_SNP[valIdx], PHENOTYPE[valIdx]
		testSNP, testSNP_QA, testPheno = imp_SNP[testIdx], QA_SNP[testIdx], PHENOTYPE[testIdx]

		# Train and evaluate on imputed data
		history, corr = model_train(
			testSNP, valSNP, trainSNP, testPheno, valPheno, trainPheno,
			f'model_IMP/model_{i}.h5',
			f'model_IMP/model_weights{i}.weights.h5'
		)
		IMP_corr.append(float(f'{corr:.4f}'))
		print(f"✅ Fold {i} (imputed) PCC: {corr:.4f}")

		# Train and evaluate on non-imputed data
		history, corr = model_train(
			testSNP_QA, valSNP_QA, trainSNP_QA, testPheno, valPheno, trainPheno,
			f'model_QA/model_{i}.h5',
			f'model_QA/model_weights{i}.weights.h5'
		)
		QA_corr.append(float(f'{corr:.4f}'))
		print(f"✅ Fold {i} (non-imputed) PCC: {corr:.4f}")

		# 🧹 Clear memory
		from keras import backend as K
		import gc
		K.clear_session()
		gc.collect()

	if run_fold is not None:
		with open("fold_pcc_log.csv", "a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow([run_fold, IMP_corr[0], QA_corr[0]])
			print(f"✅ Saved fold {run_fold} to fold_pcc_log.csv")

	if run_fold is None:
		with open("fold_pcc_summary.csv", mode="w", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(["Fold", "PCC_Imputed", "PCC_NonImputed"])
			for i in range(NUM_FOLDS):
				writer.writerow([i + 1, IMP_corr[i], QA_corr[i]])

if __name__ == '__main__':
	
	#os.chdir("MOISTURE")

	parser = argparse.ArgumentParser()
	parser.add_argument('--fold', type=int, default=None, help="Fold number (1–10)")
	parser.add_argument('--summary', action='store_true', help="Run saliency summary after all folds")
	args = parser.parse_args()

	IMP_input = "IMP_height.txt"
	QA_input = "QA_height.txt"

	assign_folds_to_file("IMP_height.txt")  # assign new folds
	sync_folds_column("IMP_height.txt", "QA_height.txt")  # sync folds to QA

	if args.summary:
		run_saliency_summary(IMP_input, QA_input)
	elif args.fold:
		main(IMP_input, QA_input, run_fold=args.fold)
	else:
		print("🌀 No fold specified — running all folds via run_all_folds.py ...")
		subprocess.run(['python3', 'run_all_folds.py'])
