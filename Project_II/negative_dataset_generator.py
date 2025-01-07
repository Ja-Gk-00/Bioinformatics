import random
import pandas as pd
from Bio import SeqIO

genome_fasta = "Data/GRCh38.fa"        
positive_file = "Data/positives.txt"   
output_file = "Data/random_negatives.tsv"  

with open(positive_file, 'r') as f:
    positive_seqs = [line.strip() for line in f if line.strip()]

positive_lengths = [len(seq) for seq in positive_seqs]

genome = {}
for record in SeqIO.parse(genome_fasta, "fasta"):
    genome[record.id] = str(record.seq).upper()

num_random = len(positive_seqs)
random_negatives = []
attempts = 0
max_attempts = 10_000_000

chrom_list = list(genome.keys())

while len(random_negatives) < num_random and attempts < max_attempts:
    attempts += 1
    chrom = random.choice(chrom_list)
    length = random.choice(positive_lengths)
    chrom_len = len(genome[chrom])
    if length > chrom_len:
        continue
    start = random.randint(1, chrom_len - length + 1)
    end = start + length - 1
    seq = genome[chrom][start-1:end]
    if 'N' in seq:
        continue
    random_negatives.append(seq)

if len(random_negatives) < num_random:
    raise RuntimeError("Could not generate enough random negative sequences.")

df_random_negatives = pd.DataFrame({
    "sequence": random_negatives,
    "label": [0]*len(random_negatives)
})

df_random_negatives.to_csv(output_file, sep='\t', index=False)
print(f"Random negative dataset saved to: {output_file}")
