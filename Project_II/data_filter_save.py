import pandas as pd
import random
from Bio import SeqIO

experiments_file = "Data/experiments.tsv"
genome_fasta = "Data/GRCh38.fa"
positives_output = "Data/positives.txt"
negatives_vista_output = "Data/negatives_vista.txt"
negatives_random_output = "Data/negatives_random.txt"

vista = pd.read_csv(experiments_file, sep='\t')

positive = vista[vista["curation_status"] == "positive"].copy()
negative_vista = vista[vista["curation_status"] == "negative"].copy()

def parse_coordinate(coord_str):
    chrom, positions = coord_str.split(':')
    start, end = positions.split('-')
    return chrom, int(start), int(end)

positive = positive.dropna(subset=["coordinate_hg38"])
positive[['chrom','start','end']] = positive['coordinate_hg38'].apply(lambda x: pd.Series(parse_coordinate(str(x))))

negative_vista = negative_vista.dropna(subset=["coordinate_hg38"])
negative_vista[['chrom','start','end']] = negative_vista['coordinate_hg38'].apply(lambda x: pd.Series(parse_coordinate(x)))


with open(positives_output, 'w') as f:
    for seq in positive['seq_hg38']:
        f.write(seq.upper() + "\n")

with open(negatives_vista_output, 'w') as f:
    for seq in negative_vista['seq_hg38']:
        f.write(seq.upper() + "\n")

genome = {}
for record in SeqIO.parse(genome_fasta, "fasta"):
    genome[record.id] = str(record.seq).upper()

pos_intervals = {}
for chrom in positive['chrom'].unique():
    chrom_intervals = positive[positive['chrom'] == chrom][['start','end']].values.tolist()
    chrom_intervals.sort(key=lambda x: x[0]) 
    pos_intervals[chrom] = chrom_intervals

def overlaps_with_positive(chrom, start, end):
    intervals = pos_intervals.get(chrom, [])
    left, right = 0, len(intervals)-1
    while left <= right:
        mid = (left + right) // 2
        s, e = intervals[mid]
        if start > e:
            left = mid + 1
        elif end < s:
            right = mid - 1
        else:
            return True
    return False

num_random = len(positive)
positive_lengths = positive.apply(lambda row: row['end'] - row['start'] + 1, axis=1).tolist()
chrom_list = list(genome.keys())
random_negatives = []

attempts = 0
max_attempts = 10_000_000

while len(random_negatives) < num_random and attempts < max_attempts:
    attempts += 1
    chrom = random.choice(chrom_list)
    length = random.choice(positive_lengths)
    chrom_len = len(genome[chrom])

    if length > chrom_len:
        continue

    start = random.randint(1, chrom_len - length + 1)
    end = start + length - 1

    if overlaps_with_positive(chrom, start, end):
        continue

    seq = genome[chrom][start-1:end]
    if 'N' in seq:
        continue

    random_negatives.append(seq)

if len(random_negatives) < num_random:
    raise RuntimeError("Could not generate enough random negative sequences without overlap and without N.")

# Save
with open(negatives_random_output, 'w') as f:
    for seq in random_negatives:
        f.write(seq.upper() + "\n")

print("Positive sequences saved to:", positives_output)
print("Negative VISTA sequences saved to:", negatives_vista_output)
print("Random negative sequences saved to:", negatives_random_output)
