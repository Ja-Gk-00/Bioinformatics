import pandas as pd
from Bio.Seq import Seq
import itertools

positives_file = "Data/positives.txt"
negatives_vista_file = "Data/negatives_vista.txt"
negatives_random_file = "Data/negatives_random.txt"

with open(positives_file, 'r') as f:
    positive_seqs = [line.strip() for line in f if line.strip()]

with open(negatives_vista_file, 'r') as f:
    negative_vista_seqs = [line.strip() for line in f if line.strip()]

with open(negatives_random_file, 'r') as f:
    negative_random_seqs = [line.strip() for line in f if line.strip()]

def generate_kmer_feature_list(k):
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    seen = set()
    features = []
    for kmer in all_kmers:
        rc = str(Seq(kmer).reverse_complement())
        if kmer not in seen:
            if kmer == rc:
                features.append(kmer)
                seen.add(kmer)
            else:
                rep = min(kmer, rc)
                features.append(rep)
                seen.add(kmer)
                seen.add(rc)
    return sorted(features)

def count_kmers(seq, k, feature_list):
    length = len(seq)
    if length < k:
        return [0.0]*len(feature_list)
    
    kmer_counts = {f:0 for f in feature_list}
    for i in range(length - k + 1):
        kmer = seq[i:i+k]
        rc = str(Seq(kmer).reverse_complement())
        rep = min(kmer, rc)
        if rep in kmer_counts:
            kmer_counts[rep] += 1
    
    freq_vector = [c/length for c in kmer_counts.values()]
    return freq_vector

def build_feature_df(sequences, label, k, feature_list):
    data = [count_kmers(seq, k, feature_list) for seq in sequences]
    df = pd.DataFrame(data, columns=feature_list)
    df['label'] = label
    return df

k_values = [3,4,5,6,7,8,9,10]

for k in k_values:
    feature_list = generate_kmer_feature_list(k)

    positive_df = build_feature_df(positive_seqs, 1, k, feature_list)
    negative_vista_df = build_feature_df(negative_vista_seqs, 0, k, feature_list)
    negative_random_df = build_feature_df(negative_random_seqs, 0, k, feature_list)

    # Save
    positive_df.to_csv(f"positive_{k}_mer.csv", index=False)
    negative_vista_df.to_csv(f"negative_vista{k}_mer.csv", index=False)
    negative_random_df.to_csv(f"negative_random{k}_mer.csv", index=False)

    print(f"K={k}:")
    print("Positive DataFrame shape:", positive_df.shape)
    print("Vista Negative DataFrame shape:", negative_vista_df.shape)
    print("Random Negative DataFrame shape:", negative_random_df.shape)
    print("Number of features:", len(feature_list))
    print("-"*40)
