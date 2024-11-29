import random
import numpy as np

def calculate_probability_distribution(sequence, confidence_values):
    """Returns a list of dictionaries where each dictionary contains nucleotides as keys and their probabilities as values"""
    nucleotide_probs = []
    
    for i, (nucleotide, confidence) in enumerate(zip(sequence, confidence_values)):
        probabilities = {nuc: 0.0 for nuc in "ACGT"}
        probabilities[nucleotide] = confidence
        
        remaining_prob = (1.0 - confidence) / 3
        for other_nuc in "ACGT":
            if other_nuc != nucleotide:
                probabilities[other_nuc] = remaining_prob
        
        nucleotide_probs.append(probabilities)
    
    return nucleotide_probs

def sample_from_probabilistic_genome(probabilistic_genome, query_length):
    """Generates a query sequence by sampling nucleotides based on the probability distributions from a probabilistic genome."""
    start_pos = random.randint(0, len(probabilistic_genome) - query_length)
    query_sequence = ""

    for i in range(start_pos, start_pos + query_length):
        probabilities = probabilistic_genome[i]
        nucleotides = list(probabilities.keys())
        probs = list(probabilities.values())
        chosen_nucleotide = np.random.choice(nucleotides, p=probs) #weighted random choice
        query_sequence += chosen_nucleotide

    return query_sequence

def create_kmer(query_sequence, k):
    """Creates kmers of length k from a query sequence"""
    kmers = []
    for i in range(len(query_sequence) - k + 1):
        kmers.append(query_sequence[i:i + k])
    return kmers


def main():
    with open("genomesequence.fa", "r") as f_file:
        sequence = f_file.readline().strip() 

    with open("chr22.maf.ancestors.42000000.complete.boreo.conf.txt", "r") as p_file:
        confidence_values = list(map(float, p_file.readline().split()))  

    prob_distributions = calculate_probability_distribution(sequence, confidence_values)

    for i, probs in enumerate(prob_distributions):
        print(f"Position {i + 1}: {probs}")

    query_length = 10  #this can change
    
    print("\nGenerated Query Sequences:")
    for _ in range(5):  #5 queries, we can change to only do one at a time
        query_sequence = sample_from_probabilistic_genome(prob_distributions, query_length)
        print(f"Query Sequence: {query_sequence}")

    k=5 #subject to change
    print("\nKmers:")
    for _ in range(5):
        query_sequence = sample_from_probabilistic_genome(prob_distributions, query_length)
        kmers = create_kmer(query_sequence, k)
        print(f"Query Sequence: {query_sequence}")
        print(f"Kmers: {kmers}\n")


if __name__ == "__main__":
    main()
