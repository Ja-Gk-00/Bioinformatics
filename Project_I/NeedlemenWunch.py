import numpy as np
import pandas as pd
import tabulate

DNAfull = pd.DataFrame(
    [[5, -4, -4, -1],
     [-4, 5, -4, -1],
     [-4, -4, 5, -1],
     [-1, -1, -1, 5]],
    index=['A', 'G', 'C', 'T'],
    columns=['A', 'G', 'C', 'T']
)

def needleman_wunsch(seq1, seq2, gap_penalty=-2, n=1, print_matrix = True ,output_filename="nw_output.txt"):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1))
    
    for i in range(m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(n + 1):
        dp[0][j] = j * gap_penalty
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + DNAfull.loc[seq1[i-1], seq2[j-1]]
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)
    
    if print_matrix:
        print("DP Matrix:")
        headers = ["-"] + list(seq2)
        dp_table = [["-"] + list(dp[0])]
        for i in range(1, m + 1):
            dp_table.append([seq1[i - 1]] + list(dp[i]))
        print(tabulate.tabulate(dp_table, headers, tablefmt="grid"))

    
    alignments = []
    def traceback(i, j, aligned_seq1, aligned_seq2):
        if len(alignments) >= n:
            return
        if i == 0 and j == 0:
            alignments.append((aligned_seq1[::-1], aligned_seq2[::-1]))
            return
        if i > 0 and dp[i][j] == dp[i-1][j] + gap_penalty:
            traceback(i-1, j, aligned_seq1 + seq1[i-1], aligned_seq2 + '-')
        if j > 0 and dp[i][j] == dp[i][j-1] + gap_penalty:
            traceback(i, j-1, aligned_seq1 + '-', aligned_seq2 + seq2[j-1])
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + DNAfull.loc[seq1[i-1], seq2[j-1]]:
            traceback(i-1, j-1, aligned_seq1 + seq1[i-1], aligned_seq2 + seq2[j-1])
    
    traceback(m, n, '', '')
    
    with open(output_filename, 'w') as f:
        for idx, (aligned_seq1, aligned_seq2) in enumerate(alignments):
            f.write(f"Global alignment no. {idx + 1}:\n")
            f.write(f"{aligned_seq1}\n{aligned_seq2}\n")
            f.write(f"Score: {dp[m][n]}\n\n")

def smith_waterman(seq1, seq2, gap_penalty=-2, print_matrix = True, output_filename="sw_output.txt"):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1))
    
    max_score = 0
    max_pos = None
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + DNAfull.loc[seq1[i-1], seq2[j-1]]
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(0, match, delete, insert)
            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_pos = (i, j)
    
    aligned_seq1, aligned_seq2 = '', ''
    i, j = max_pos
    while i > 0 and j > 0 and dp[i][j] != 0:
        if dp[i][j] == dp[i-1][j-1] + DNAfull.loc[seq1[i-1], seq2[j-1]]:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] + gap_penalty:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = '-' + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = '-' + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1
    
    if print_matrix:
        print("DP Matrix:")
        headers = ["-"] + list(seq2)
        dp_table = [["-"] + list(dp[0])]
        for i in range(1, m + 1):
            dp_table.append([seq1[i - 1]] + list(dp[i]))
        print(tabulate.tabulate(dp_table, headers, tablefmt="grid"))


    with open(output_filename, 'w') as f:
        f.write(f"Local alignment:\n")
        f.write(f"{aligned_seq1}\n{aligned_seq2}\n")
        f.write(f"Score: {max_score}\n")
