from NeedlemenWunch import needleman_wunsch, smith_waterman

if __name__ == "__main__":
    seq1 = input("Enter the first DNA sequence: ")
    seq2 = input("Enter the second DNA sequence: ")
    gap_penalty = int(input("Enter the gap penalty (default -2): ") or -2)
    n = int(input("Enter the number of optimal alignments to find: "))
    
    algorithm = input("Which algorithm would you like to run? (Needleman-Wunsch = 0 / Smith-Waterman = 1): ")
    
    if algorithm == "0":
        print("\nRunning Needleman-Wunsch Algorithm...")
        needleman_wunsch(seq1, seq2, gap_penalty=gap_penalty, n=n, output_filename="Data/nw_output.txt")
    elif algorithm == "1":
        print("\nRunning Smith-Waterman Algorithm...")
        smith_waterman(seq1, seq2, gap_penalty=gap_penalty, output_filename="Data/sm_output.txt")
    else:
        print("Invalid algorithm choice. Please choose either 'Needleman-Wunsch' or 'Smith-Waterman'.")
