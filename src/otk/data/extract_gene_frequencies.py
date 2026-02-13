import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

"""
Extract gene level frequency features from modeling data and store them in a gene-specific file.
"""

def extract_gene_frequencies(data_path, output_path):
    """
    Extract gene level frequency features from modeling data.
    
    Args:
        data_path (str): Path to the modeling data CSV file
        output_path (str): Path to save the gene frequency data
    """
    # Load the modeling data
    logger.info(f"Loading modeling data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded with shape: {df.shape}")
    
    # Extract gene level frequency features
    gene_freq_cols = ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']
    
    # Group by gene_id and calculate the mean for each frequency feature
    # Using mean as the aggregation method - this assumes the frequencies are consistent per gene
    gene_freqs = df.groupby('gene_id')[gene_freq_cols].mean().reset_index()
    
    logger.info(f"Extracted gene frequencies for {len(gene_freqs)} genes")
    logger.info(f"Sample gene frequencies:")
    logger.info(gene_freqs.head().to_string())
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the gene frequencies to a CSV file
    gene_freqs.to_csv(output_path, index=False)
    logger.info(f"Gene frequencies saved to {output_path}")

if __name__ == "__main__":
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(project_root))), "data", "gcap_modeling_data.csv.gz")
    output_path = os.path.join(project_root, "data", "gene_frequencies.csv")
    
    # Extract gene frequencies
    extract_gene_frequencies(data_path, output_path)
