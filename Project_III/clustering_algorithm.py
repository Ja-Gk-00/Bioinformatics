import os
import glob
import re
import subprocess
import tempfile
from Bio import SeqIO, pairwise2, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import ListedColormap, to_hex, to_rgb

# Path to the MUSCLE executable
MUSCLE_PATH = 'muscle'  # Ensure MUSCLE is in your PATH or provide the full path

def assign_colors(categories):
    """
    Assigns unique colors to each category for visualization purposes.
    """
    unique_categories = sorted(set(categories))
    num_categories = len(unique_categories)
    cmap = plt.cm.get_cmap('tab20', num_categories)
    color_dict = {category: to_hex(cmap(i)) for i, category in enumerate(unique_categories)}
    return color_dict

def parse_fasta_from_directory(directory):
    """
    Parses all FASTA files in the specified directory and extracts sequences along with metadata.
    """
    sequences = []
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in directory '{directory}'.")
        return sequences
    for file in txt_files:
        with open(file, 'r') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                description = record.description
                # Extract Animal from description, assuming format [Animal]
                animal_match = re.search(r'\[(.*?)\]', description)
                animal = animal_match.group(1) if animal_match else "Unknown"
                # Extract Protein group from description, assuming it follows the first word
                protein_match = re.search(r'^\S+\s+(.+?)\s+\[', description)
                protein = protein_match.group(1) if protein_match else "Unknown"
                sequences.append({
                    'Sequence_ID': record.id,
                    'Description': description,
                    'Protein': protein,
                    'Animal': animal,
                    'Sequence': str(record.seq)
                })
    return sequences

def compute_similarity_matrix(sequences):
    """
    Computes a pairwise similarity matrix based on global pairwise alignment.
    """
    ids = [seq['Sequence_ID'] for seq in sequences]
    n = len(ids)
    similarity_matrix = np.zeros((n, n))
    print("Computing pairwise similarities:")
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity = 100.0
            else:
                alignments = pairwise2.align.globalxx(sequences[i]['Sequence'], sequences[j]['Sequence'], one_alignment_only=True)
                if alignments:
                    aln1, aln2, score, start, end = alignments[0]
                    matches = score
                    length = max(len(sequences[i]['Sequence']), len(sequences[j]['Sequence']))
                    similarity = (matches / length) * 100
                else:
                    similarity = 0.0
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
        print(f"Computed similarities for sequence {i+1}/{n}")
    return similarity_matrix, ids

def create_distance_matrix(similarity_matrix):
    """
    Converts similarity scores to distances.
    """
    distance_matrix = 100 - similarity_matrix
    return distance_matrix

def hierarchical_clustering(distance_matrix, method='average'):
    """
    Performs hierarchical clustering using the specified linkage method.
    """
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method=method)
    return linkage_matrix

def perform_msa(sequences_subset, output_format='fasta'):
    """
    Performs Multiple Sequence Alignment (MSA) using MUSCLE.
    Returns the alignment file path if successful, else None.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix='.fasta') as temp_input:
        SeqIO.write(sequences_subset, temp_input, "fasta")
        temp_input_name = temp_input.name

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', suffix='.fasta') as temp_output:
        temp_output_name = temp_output.name

    muscle_cmd = f"{MUSCLE_PATH} -in {temp_input_name} -out {temp_output_name}"
    try:
        subprocess.run(muscle_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(temp_output_name) and os.path.getsize(temp_output_name) > 0:
            return temp_output_name
        else:
            print(f"MUSCLE failed to produce output for subset with {len(sequences_subset)} sequences.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"MUSCLE execution failed: {e}")
        return None

def build_upgma_tree(alignment_file):
    """
    Builds a UPGMA tree from the given alignment file.
    Returns the tree object if successful, else None.
    """
    try:
        alignment = SeqIO.read(alignment_file, "fasta")
    except Exception as e:
        print(f"Failed to read alignment file '{alignment_file}': {e}")
        return None

    try:
        calculator = DistanceCalculator('identity')
        distance_matrix = calculator.get_distance(alignment)
    except Exception as e:
        print(f"Error computing distance matrix from alignment: {e}")
        return None

    try:
        constructor = DistanceTreeConstructor()
        tree = constructor.upgma(distance_matrix)
        return tree
    except Exception as e:
        print(f"Error constructing UPGMA tree: {e}")
        return None

def visualize_tree(tree, color_map, title, filename):
    """
    Visualizes and saves the phylogenetic tree with colored branches.
    """
    if tree is None:
        print(f"No tree to visualize for '{title}'.")
        return

    # Assign colors to terminal nodes based on the color_map
    for clade in tree.get_terminals():
        clade.color = color_map.get(clade.name, "#000000")  # Default to black if not found

    # Draw the tree
    plt.figure(figsize=(20, 10))
    Phylo.draw(
        tree,
        branch_labels=lambda c: None,
        do_show=False,
        show_confidence=False,
        label_colors=None
    )

    # Assign colors to terminal labels
    ax = plt.gca()
    for terminal in tree.get_terminals():
        label = terminal.name
        color = color_map.get(label.split(' - ')[0], "#000000")  # Assuming color based on the first part (Animal or Protein)
        # The following code attempts to color the text labels; however, Biopython's Phylo.draw does not provide direct access to label artists
        # As a workaround, you can customize the drawing function or use other libraries like ete3 for more control
        # For simplicity, we'll skip manual coloring here

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Phylogenetic tree '{title}' saved as '{filename}'.")

def plot_dendrogram(linkage_matrix, labels, metadata, cluster_num, title='Hierarchical Clustering Dendrogram', figsize=(20, 10), filename=None):
    """
    Plots and saves the dendrogram with colored labels.
    """
    plt.figure(figsize=figsize)
    dendro = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=None 
    )
    animal_colors = assign_colors([meta['Animal'] for meta in metadata])
    protein_colors = assign_colors([meta['Protein'] for meta in metadata])
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        try:
            animal, protein = lbl.get_text().split(' - ')
            animal_color = np.array(to_rgb(animal_colors.get(animal, '#000000')))
            protein_color = np.array(to_rgb(protein_colors.get(protein, '#000000')))
            blended_color = (animal_color + protein_color) / 2
            lbl.set_color(to_hex(blended_color))
        except ValueError:
            # In case the label doesn't have both animal and protein
            lbl.set_color('#000000')
    plt.title(title)
    plt.xlabel('Sequence ID')
    plt.ylabel('Distance (100 - Similarity)')
    plt.tight_layout()
    
    # Determine filename
    if filename is None:
        dendro_filename = f'dendrogram_{cluster_num}_clusters.png'
    else:
        dendro_filename = filename
    
    plt.savefig(dendro_filename, dpi=300)
    plt.close()
    print(f"Dendrogram saved as '{dendro_filename}'.")

def determine_optimal_clusters(distance_matrix, range_min=2, range_max=10):
    """
    Determines the optimal number of clusters using silhouette scores.
    """
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    n_samples = distance_matrix.shape[0]
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(distance_matrix)
    best_score = -1
    best_k = 2
    for k in range(range_min, min(range_max, n_samples)):
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
        if len(set(cluster_labels)) == 1:
            print(f"All sequences assigned to one cluster for k={k}. Skipping silhouette score.")
            continue
        score = silhouette_score(pca_features, cluster_labels)
        print(f"Silhouette Score for {k} clusters: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Optimal number of clusters based on Silhouette Score: {best_k}")
    return best_k

def plot_tsne(similarity_matrix, labels, metadata, clusters, title='t-SNE Visualization of Protein Clusters', figsize=(12, 8)):
    """
    Plots and saves a t-SNE visualization of the clusters.
    """
    distance_matrix = create_distance_matrix(similarity_matrix)
    try:
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42, perplexity=30)
        tsne_results = tsne.fit_transform(distance_matrix)
    except Exception as e:
        print(f"t-SNE failed: {e}")
        return
    df_tsne = pd.DataFrame({
        'TSNE1': tsne_results[:,0],
        'TSNE2': tsne_results[:,1],
        'Cluster': clusters,
        'Animal': [meta['Animal'] for meta in metadata],
        'Protein': [meta['Protein'] for meta in metadata],
        'Label': [f"{meta['Animal']} - {meta['Protein']}" for meta in metadata]
    })
    animal_colors = assign_colors(df_tsne['Animal'])
    protein_colors = assign_colors(df_tsne['Protein'])
    label_colors = []
    for animal, protein in zip(df_tsne['Animal'], df_tsne['Protein']):
        animal_color = np.array(to_rgb(animal_colors[animal]))
        protein_color = np.array(to_rgb(protein_colors[protein]))
        blended_color = (animal_color + protein_color) / 2
        label_colors.append(blended_color)
    plt.figure(figsize=figsize)
    scatter = sns.scatterplot(
        x='TSNE1', y='TSNE2',
        hue='Cluster',
        palette='tab10',
        data=df_tsne,
        legend='full',
        alpha=0.7
    )
    # Annotate points
    texts = []
    for i in range(df_tsne.shape[0]):
        texts.append(
            plt.text(
                df_tsne['TSNE1'][i],
                df_tsne['TSNE2'][i],
                f"{df_tsne['Animal'][i]} - {df_tsne['Protein'][i]}",
                fontsize=8,
                color=to_hex(label_colors[i]),
                weight='bold'
            )
        )
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey', lw=0.5))
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Extract cluster number from title
    try:
        cluster_num = re.search(r'(\d+) Clusters', title).group(1)
    except:
        cluster_num = 'unknown'
    tsne_filename = f'tsne_clusters_{cluster_num}_clusters.png'
    plt.savefig(tsne_filename, dpi=300)
    plt.close()
    print(f"t-SNE plot saved as '{tsne_filename}'.")

def plot_heatmap(similarity_matrix, clusters, labels, cluster_num, title='Cluster Heatmap', figsize=(12, 10)):
    """
    Plots and saves a heatmap of the similarity matrix.
    """
    sim_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    cluster_order = pd.Series(clusters, index=labels).sort_values().index
    plt.figure(figsize=figsize)
    sns.clustermap(
        sim_df.loc[cluster_order, cluster_order],
        row_cluster=False,
        col_cluster=False,
        cmap='viridis',
        linewidths=.5,
        figsize=(15, 15)
    )
    plt.title(title)
    heatmap_filename = f'cluster_heatmap_{cluster_num}_clusters.png'
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()
    print(f"Cluster heatmap saved as '{heatmap_filename}'.")

def assign_clusters(linkage_matrix, num_clusters):
    """
    Assigns cluster labels based on the linkage matrix and the desired number of clusters.
    """
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    return cluster_labels

def save_similarity_matrix(similarity_matrix, ids, filename='similarity_matrix.csv'):
    """
    Saves the similarity matrix to a CSV file.
    """
    sim_df = pd.DataFrame(similarity_matrix, index=ids, columns=ids)
    sim_df.to_csv(filename)
    print(f"Similarity matrix saved to '{filename}'.")

def save_cluster_assignments(sequence_data, cluster_labels, filename='cluster_assignments.csv'):
    """
    Saves the cluster assignments to a CSV file.
    """
    df = pd.DataFrame(sequence_data)
    df['Cluster'] = cluster_labels
    df.to_csv(filename, index=False)
    print(f"Cluster assignments saved to '{filename}'.")

def main():
    """
    Main function to orchestrate the phylogenetic analysis.
    """
    directory = 'Data_FASTA/'  # Directory containing FASTA files
    silhouette_range_min = 2
    silhouette_range_max = 10
    cluster_numbers = range(silhouette_range_min, silhouette_range_max + 1)
    
    print("Parsing FASTA sequences from directory...")
    sequences = parse_fasta_from_directory(directory)
    if not sequences:
        print("No sequences found. Exiting.")
        return
    print(f"Total sequences parsed: {len(sequences)}")
    
    print("\nComputing similarity matrix...")
    similarity_matrix, ids = compute_similarity_matrix(sequences)
    save_similarity_matrix(similarity_matrix, ids, filename='similarity_matrix.csv')
    
    print("\nCreating distance matrix...")
    distance_matrix = create_distance_matrix(similarity_matrix)
    
    print("\nPerforming hierarchical clustering...")
    linkage_matrix = hierarchical_clustering(distance_matrix, method='average')
    
    print("\nDetermining the optimal number of clusters using Silhouette Score...")
    optimal_clusters = determine_optimal_clusters(distance_matrix, range_min=silhouette_range_min, range_max=silhouette_range_max)
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    print("\nAssigning clusters based on optimal number...")
    cluster_labels = assign_clusters(linkage_matrix, optimal_clusters)
    save_cluster_assignments(sequences, cluster_labels, filename=f'cluster_assignments_{optimal_clusters}_clusters.csv')
    
    # Prepare labels for dendrogram and other plots
    labels = [f"{seq['Animal']} - {seq['Protein']}" for seq in sequences]
    
    print("\nPlotting full dendrogram...")
    plot_dendrogram(
        linkage_matrix, 
        labels, 
        sequences, 
        cluster_num=optimal_clusters, 
        title=f'Hierarchical Clustering Dendrogram ({optimal_clusters} Clusters)'
    )
    
    print("\nPlotting t-SNE visualization...")
    plot_tsne(
        similarity_matrix, 
        labels, 
        sequences, 
        cluster_labels, 
        title=f't-SNE Visualization of Protein Clusters ({optimal_clusters} Clusters)'
    )
    
    print("\nPlotting heatmap...")
    plot_heatmap(
        similarity_matrix, 
        cluster_labels, 
        labels, 
        cluster_num=optimal_clusters, 
        title=f'Cluster Heatmap ({optimal_clusters} Clusters)'
    )
    
    # Building separate dendrograms
    print("\nBuilding phylogenetic trees...")
    
    # i. Separate dendrograms for each organism
    print("\nBuilding dendrograms for each organism...")
    organisms = sorted(set(seq['Animal'] for seq in sequences))
    for organism in organisms:
        subset_seqs = [seq for seq in sequences if seq['Animal'] == organism]
        if len(subset_seqs) < 2:
            print(f"Not enough sequences for organism '{organism}' to build a dendrogram. Skipping.")
            continue
        subset_ids = [seq['Sequence_ID'] for seq in subset_seqs]
        subset_labels = [f"{seq['Animal']} - {seq['Protein']}" for seq in subset_seqs]
        subset_indices = [sequences.index(seq) for seq in subset_seqs]
        subset_similarity = similarity_matrix[np.ix_(subset_indices, subset_indices)]
        subset_distance = create_distance_matrix(subset_similarity)
        subset_linkage = hierarchical_clustering(subset_distance, method='average')
        # Plot dendrogram for this organism
        dendro_filename = f'dendrogram_{optimal_clusters}_clusters_organism_{organism}.png'
        plot_dendrogram(
            subset_linkage,
            subset_labels,
            subset_seqs,
            cluster_num=optimal_clusters,
            title=f'Hierarchical Clustering Dendrogram for Organism: {organism} ({optimal_clusters} Clusters)',
            filename=dendro_filename
        )
    
    # ii. Separate dendrograms for each protein group
    print("\nBuilding dendrograms for each protein group...")
    proteins = sorted(set(seq['Protein'] for seq in sequences))
    for protein in proteins:
        subset_seqs = [seq for seq in sequences if seq['Protein'] == protein]
        if len(subset_seqs) < 2:
            print(f"Not enough sequences for protein group '{protein}' to build a dendrogram. Skipping.")
            continue
        subset_ids = [seq['Sequence_ID'] for seq in subset_seqs]
        subset_labels = [f"{seq['Animal']} - {seq['Protein']}" for seq in subset_seqs]
        subset_indices = [sequences.index(seq) for seq in subset_seqs]
        subset_similarity = similarity_matrix[np.ix_(subset_indices, subset_indices)]
        subset_distance = create_distance_matrix(subset_similarity)
        subset_linkage = hierarchical_clustering(subset_distance, method='average')
        # Plot dendrogram for this protein group
        dendro_filename = f'dendrogram_{optimal_clusters}_clusters_protein_{protein}.png'
        plot_dendrogram(
            subset_linkage,
            subset_labels,
            subset_seqs,
            cluster_num=optimal_clusters,
            title=f'Hierarchical Clustering Dendrogram for Protein Group: {protein} ({optimal_clusters} Clusters)',
            filename=dendro_filename
        )
    
    # iii. Separate dendrograms for each cluster
    print("\nBuilding dendrograms for each cluster...")
    unique_clusters = sorted(set(cluster_labels))
    for unique_cluster in unique_clusters:
        subset_seqs = [seq for seq, label in zip(sequences, cluster_labels) if label == unique_cluster]
        if len(subset_seqs) < 2:
            print(f"Not enough sequences in cluster {unique_cluster} to build a dendrogram. Skipping.")
            continue
        subset_ids = [seq['Sequence_ID'] for seq in subset_seqs]
        subset_labels = [f"{seq['Animal']} - {seq['Protein']}" for seq in subset_seqs]
        subset_indices = [sequences.index(seq) for seq in subset_seqs]
        subset_similarity = similarity_matrix[np.ix_(subset_indices, subset_indices)]
        subset_distance = create_distance_matrix(subset_similarity)
        subset_linkage = hierarchical_clustering(subset_distance, method='average')
        # Plot dendrogram for this cluster
        dendro_filename = f'dendrogram_{optimal_clusters}_clusters_specific_cluster_{unique_cluster}.png'
        plot_dendrogram(
            subset_linkage,
            subset_labels,
            subset_seqs,
            cluster_num=optimal_clusters,
            title=f'Hierarchical Clustering Dendrogram for Cluster {unique_cluster} ({optimal_clusters} Clusters)',
            filename=dendro_filename
        )
    
    # iv. One common dendrogram for all sequences (already plotted above as full dendrogram)
    # If you want to plot it again specifically, you can uncomment the following lines:
    """
    print("\nRe-building common dendrogram for all sequences...")
    plot_dendrogram(
        linkage_matrix,
        labels,
        sequences,
        cluster_num=optimal_clusters,
        title=f'Common Hierarchical Clustering Dendrogram ({optimal_clusters} Clusters)',
        filename=f'dendrogram_{optimal_clusters}_clusters_all_sequences.png'
    )
    """
    
    print("\nPhylogenetic analysis completed successfully.")

if __name__ == "__main__":
    main()
