import unittest
from NeedlemenWunch import needleman_wunsch, smith_waterman

class TestSequenceAlignment(unittest.TestCase):
    def test_needleman_wunsch(self):
        seq1 = "GATTACA"
        seq2 = "GCATGCT"
        needleman_wunsch(seq1, seq2, n=1, output_filename="test_nw_output.txt")
        with open("test_nw_output.txt", 'r') as f:
            result = f.read()
        self.assertIn("Global alignment no. 1:", result)
        self.assertIn("Score:", result)

    def test_smith_waterman(self):
        seq1 = "GATTACA"
        seq2 = "GCATGCG"
        smith_waterman(seq1, seq2, output_filename="test_sw_output.txt")
        with open("test_sw_output.txt", 'r') as f:
            result = f.read()
        self.assertIn("Local alignment:", result)
        self.assertIn("Score:", result)

    def test_nw_correctness(self):
        seq1 = "ATCG"
        seq2 = "ATG"
        needleman_wunsch(seq1, seq2, n=1, output_filename="test_nw_correctness.txt")
        with open("test_nw_correctness.txt", 'r') as f:
            result = f.read()
        self.assertIn("ATCG\nAT-G", result)

    def test_long_needleman_wunsch(self):
        seq1 = "ACGTGCTAGCTAGTACGATCGATGCTAGCTGATCGTAGCTG"
        seq2 = "TGCATGCATGCGTAGCTAGCTGATCGATCGTAGCTAGCTAG"
        needleman_wunsch(seq1, seq2, n=1, output_filename="test_long_nw_output.txt")
        with open("test_long_nw_output.txt", 'r') as f:
            result = f.read()
        self.assertIn("Global alignment no. 1:", result)
        self.assertIn("Score:", result)

    def test_long_smith_waterman(self):
        seq1 = "ACGTGCTAGCTAGTACGATCGATGCTAGCTGATCGTAGCTG"
        seq2 = "TGCATGCATGCGTAGCTAGCTGATCGATCGTAGCTAGCTAG"
        smith_waterman(seq1, seq2, output_filename="test_long_sw_output.txt")
        with open("test_long_sw_output.txt", 'r') as f:
            result = f.read()
        self.assertIn("Local alignment:", result)
        self.assertIn("Score:", result)

if __name__ == "__main__":
    unittest.main()
