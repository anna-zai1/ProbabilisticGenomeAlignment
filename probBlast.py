import random
import numpy as np
from collections import defaultdict
import collections
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

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

def calculate_majority_sequence(probabilistic_genome):
    """Construct the majority sequence by taking the most probable nucleotide at each position."""
    majority_sequence = ""
    for position_probs in probabilistic_genome:
        majority_nuc = max(position_probs, key=position_probs.get)  # Nucleotide with highest probability
        majority_sequence += majority_nuc
    return majority_sequence

def sample_from_probabilistic_genome(probabilistic_genome, query_length):
    """Generates a query sequence by sampling nucleotides based on the probability distributions from a probabilistic genome."""
    start_pos = random.randint(0, len(probabilistic_genome) - query_length)
    query_sequence = ""

    for i in range(start_pos, start_pos + query_length):
        probabilities = probabilistic_genome[i]
        nucleotides = list(probabilities.keys())
        probs = list(probabilities.values())
        chosen_nucleotide = np.random.choice(nucleotides, p=probs)  #weighted random choice
        query_sequence += chosen_nucleotide

    return query_sequence

def create_wmer(query_sequence, k):
    """Creates wmers of length k from a query sequence"""
    wmers = []
    for i in range(len(query_sequence) - k + 1):
        wmers.append(query_sequence[i:i + k])
    return wmers

def cluster_hits(match):
    """
    Cluster adjacent hits to reduce redundant ungapped extension calculations.
    """
    for i in range(len(match) - 1):
        if match[i] and match[i + 1]:
            j = 0
            while j < len(match[i]):
                k = 0
                while k < len(match[i + 1]):
                    if match[i][j] == match[i + 1][k] - 1:
                        del match[i][j]
                        j -= 1
                        break
                    k += 1
                j += 1
    return match


def identify_hits(database, query_mers):
    """ create a list of perfect matches between wmers in the query and wmers in reference """
    match = [] # index = starting position in query, stored items = starting positions in refernce
    for i in range(0, len(query_mers) - 1):
        kmer = query_mers[i]
        hits = database.get(kmer)
        if hits is not None:
            match.append(hits)
        else:
            match.append([])

    cluster_hits(match) # cluster redundent matches
    return match

def build_kmer_database(sequence, k):
    """
    Builds a kmer database for the majority sequence
    key = kmer, value = starting postion
    If key doesn't exist return 'None'
    """
    db = collections.defaultdict(list)
    for i in range(0, len(sequence) - k -1):
        kmer = sequence[i:i + k]
        db[kmer].append(i)
    return db

def score(query_nucleotide, seq_nucleotide, prob):
    """
    Scoring for ungapped extension:
    If nucleotide matches the majority sequence: + probability(nucleotide)
    If nucleotide doesn't match the majority sequence: - complement of probability(nucleotide)
    """
    if query_nucleotide == seq_nucleotide:
        return prob.get(query_nucleotide)
    else:
        return -(1-prob.get(query_nucleotide))

def left_extension(i, j, query, seq, prob, delta):
    top_score = 0
    curr_score = 0
    while top_score - curr_score < delta and i >= 0 and j >= 0:
        curr_score += score(query[i], seq[j], prob[j])
        if curr_score > top_score:
            top_score = curr_score
        i -= 1
        j -= 1
        continue

    return top_score

def right_extension(i, j, query, seq, prob, delta):
    """
    Ungapped Extension to the right of perefect match:
    i = posotion of end of perfect match in query,
    j = posotion of end of perfect match inmajority sequence
    """
    top_score = 0
    curr_score = 0
    while top_score-curr_score < delta and i < len(query) and j < len(seq):
        curr_score += score(query[i], seq[j], prob[j]) # score
        if curr_score > top_score:
            top_score = curr_score
        i+=1
        j+=1
        continue

    return top_score

# Gapped extension

def subset_genome(q_start, ref_start, query, prob):
    """
    Returns smaller portion of reference genome to conduct NW algorithm
    (1.5 times length of query to account for possible gaps)
    q_start = index of start of perfect match in query
    ref_start = index of start of perfect match in reference genome
    """

    left_index: int = round(ref_start - (q_start * 1.5))
    right_index: int = round(ref_start + ((len(query) - q_start) * 1.5))

    # checks to avoid indexing errors:
    if left_index < 0:
        left_index = 0
    if right_index > len(prob):
        right_index = len(prob)

    short = prob[left_index:right_index]
    return short, left_index, right_index

def match_score1(query_nucleotide, genome_prob):
    """
    Scoring scheme 1:
        - Match/mismatch: + probability(nucleotide)
        - Gap: + 0
    """
    if query_nucleotide in genome_prob.keys():
        return genome_prob.get(query_nucleotide)
    else:
        return 0

def match_score2(query_nucleotide, genome_prob):
    """ Shift scale from match_score1:
    Scoring scheme 2:
        - Match/mismatch: + (probability(nucleotide) - 0.5) * 2
        - Gap: Fixed negative penalty (e.g., -1).
    """
    if query_nucleotide in genome_prob.keys():
        return (genome_prob.get(query_nucleotide) - 0.5) * 2
    else:
        return -1

def match_score3(query_nucleotide, majority_nucleotide, genome_prob):
    """
    Scoring scheme:
    - Match: + probability(nucleotide) if it matches the majority sequence nucleotide.
    - Mismatch: - (1 - probability(nucleotide)).
    - Gap: Fixed negative penalty (e.g., -1).
    """
    if query_nucleotide in genome_prob.keys():
        if query_nucleotide == majority_nucleotide:
            return genome_prob.get(query_nucleotide, 0)
        else:
            return -(1 - genome_prob.get(query_nucleotide, 0))
    else:
        return -1

def alignment(query, short, gap, ref_index, majority_sequence, scoring_function):
    """
    NW-like algorithm to perform alignment.
    Includes edge-case checks to avoid query/reference overflows.
    """

    # Initialize a DP table
    dp = np.zeros((len(query) + 1, len(short) + 1))

    # 1st row: penalize for gapping query
    for i in range(1, dp.shape[0]):
        dp[i, 0] = gap * i

    # 2nd row: no penalty (can start wherever on short)
    for j in range(1, dp.shape[1]):
        dp[0, j] = 0

    #fill in DP table
    if scoring_function == match_score1 or scoring_function == match_score2:
        for i in range(1, dp.shape[0]):
            for j in range(1, dp.shape[1]):
                dp[i, j] = max(
                    dp[i, j - 1] + gap,
                    dp[i - 1, j] + gap,
                    dp[i - 1, j - 1] + scoring_function(query[i - 1], short[j - 1])
                )
    elif scoring_function == match_score3:
        for i in range(1, dp.shape[0]):
            for j in range(1, dp.shape[1]):
                dp[i, j] = max(
                    dp[i, j - 1] + gap,
                    dp[i - 1, j] + gap,
                    dp[i, j - 1] + gap,
                    dp[i - 1, j] + gap,
                    dp[i - 1, j - 1] + scoring_function(query[i - 1], majority_sequence[ref_index + j - 1], short[j - 1])
                )

    # Retrace: start from the highest score in the last row
    i = len(query)
    j = np.argmax(dp[-1])

    ref_end = j - 1
    ref_start = ref_end
    top_score = dp[-1, j]
    aligned_query = ''
    aligned_genome = ''

    while i > 0 and j > 0:
        if j - 1 < 0:
            break

        if scoring_function == match_score1 or scoring_function == match_score2:

            if dp[i, j] == dp[i - 1, j - 1] + scoring_function(query[i - 1], short[j - 1]):
                aligned_query = query[i - 1] + aligned_query
                aligned_genome = str(max(short[j - 1], key=short[j - 1].get)) + aligned_genome
                i -= 1
                j -= 1
                ref_start -= 1

            elif dp[i, j] == dp[i, j - 1] + gap:
                aligned_query = '-' + aligned_query
                aligned_genome = str(max(short[j - 1], key=short[j - 1].get)) + aligned_genome
                j -= 1
                ref_start -= 1

            elif dp[i, j] == dp[i - 1, j] + gap:
                aligned_query = query[i - 1] + aligned_query
                aligned_genome = '-' + aligned_genome
                i -= 1

        elif scoring_function == match_score3:
            if dp[i, j] == dp[i - 1, j - 1] + scoring_function(query[i - 1], majority_sequence[ref_index + j - 1], short[j - 1]):
                aligned_query = query[i - 1] + aligned_query
                aligned_genome = str(max(short[j - 1], key=short[j - 1].get)) + aligned_genome
                i -= 1
                j -= 1
                ref_start -= 1

            elif dp[i, j] == dp[i, j - 1] + gap:
                aligned_query = '-' + aligned_query
                aligned_genome = str(max(short[j - 1], key=short[j - 1].get)) + aligned_genome
                j -= 1
                ref_start -= 1

            elif dp[i, j] == dp[i - 1, j] + gap:
                aligned_query = query[i - 1] + aligned_query
                aligned_genome = '-' + aligned_genome
                i -= 1
                ref_start -= 1

        else:
            break

    #trim any leftover gaps due to retrace limitations
    aligned_query = aligned_query.lstrip('-')
    aligned_genome = aligned_genome.lstrip('-')

    return aligned_query, aligned_genome, top_score, ref_start + ref_index, ref_end + ref_index

def call_alignment(query_sequence, prob_distributions, ungapped_scores, scoring_function, majority_sequence):
    alignments = {}
    for score, positions in ungapped_scores.items():
        for query_pos, ref_start in positions:  # Unpack positions (query_pos and ref_start)
            # Subset the genome around the alignment hit
            short, left_index, right_index = subset_genome(query_pos, ref_start, query_sequence, prob_distributions)

            # Perform alignment
            q_aligned, ref_aligned, alignment_score, ref_start_final, ref_end_final = alignment(
                query_sequence, short, -1, left_index, majority_sequence, scoring_function
            )

            # Store the result in the alignments dictionary
            alignments[(ref_start_final, ref_end_final)] = {
                "score": alignment_score,
                "query": q_aligned,
                "reference": ref_aligned,
                "ref_start": ref_start_final,
                "ref_end": ref_end_final,
            }
    return alignments

def ungapped_extension(match, query, seq, prob, delta):
    scores = collections.defaultdict(list)
    i = 0
    for i in range(0, len(match)):
        # if there is a match at that query index
        if match[i]:
            for j in range(0, len(match[i])):
                left = left_extension(i, match[i][j], query, seq, prob, delta)
                right = right_extension(i, match[i][j], query, seq, prob, delta)
                myScore = left + right
                scores[myScore].append((i, match[i][j]))
        else:
            continue

    return scores

def print_results(results, threshold):
    print("\nUngapped Extension Results:")
    print(f"{'Score':<10} {'Query Position':<15} {'Reference Position':<20} {'Gapped Extension'}")
    print("-" * 60)

    successful_gapped = 0

    for score, positions in sorted(results.items(), reverse=True):  #descending
        for query_pos, ref_pos in positions:
            gapped = "YES" if score >= threshold else "NO"
            if gapped == "YES":
                successful_gapped += 1
            print(f"{score:<10.2f} {query_pos:<15} {ref_pos:<20} {gapped}")

    print(f"\nTotal Ungapped Extensions: {sum(len(v) for v in results.values())}")
    print(f"Successful Gapped Extensions (Threshold {threshold}): {successful_gapped}")

def filter_and_sort_alignments(alignments, threshold):
    """
    Filters and sorts alignments based on a score threshold.
    """
    filtered = {
        k: v for k, v in alignments.items() if float(v['score']) >= threshold
    }

    sorted_alignments = sorted(
        filtered.items(), key=lambda x: float(x[1]['score']), reverse=True
    )

    return sorted_alignments

def evaluate_scoring(seq, confidence_values, scoring_functions):
    """
    Evaluate the scoring function on the query and reference sequences.
    """

    for scoring_function in scoring_functions:
        print(f"\nScoring Function: {scoring_function.__name__}")
        prob_distributions = calculate_probability_distribution(seq, confidence_values)
        majority_sequence = calculate_majority_sequence(prob_distributions)
        db = build_kmer_database(seq, 11)
        query = sample_from_probabilistic_genome(prob_distributions, 100)
        query_kmers = create_wmer(query, 11)
        match = identify_hits(db, query_kmers)
        ungapped_results = ungapped_extension(match, query, seq, prob_distributions, 10)
        gapped_results = call_alignment(query, prob_distributions, ungapped_results, scoring_function, majority_sequence)
        filtered_sorted_alignments = filter_and_sort_alignments(gapped_results, 15)
        #take best alignment
        best_alignment = filtered_sorted_alignments[0][1]
        aligned_query = best_alignment['query']
        ref_start = best_alignment['ref_start']
        ref_end = best_alignment['ref_end']

        #extract corresponding portion of majority sequence
        majority_subsequence = best_alignment['reference']

        precision, recall, f1 = compute_metrics(aligned_query, majority_subsequence)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


def compute_metrics(aligned_query, majority_subsequence):
    """
    Compute precision, recall, and F1 score based on positional comparison.
    """
    filtered_query = [a for a, b in zip(aligned_query, majority_subsequence) if a != '-' and b != '-']
    filtered_reference = [b for a, b in zip(aligned_query, majority_subsequence) if a != '-' and b != '-']

    true_positives = sum(1 for a, b in zip(filtered_query, filtered_reference) if a == b )
    total_predicted = sum(1 for a in filtered_query )
    total_actual = len(filtered_reference)

    precision = true_positives / total_predicted if total_predicted > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def mutate_sequence(sequence, mutation_rate):
    """Introduce mutations into the sequence at a specified rate."""
    nucleotides = "ACGT"
    mutated_seq = ""
    for nucleotide in sequence:
        if random.random() < mutation_rate:
            mutated_seq += random.choice([n for n in nucleotides if n != nucleotide])
        else:
            mutated_seq += nucleotide
    return mutated_seq

def evaluate_single_run(args):
    """Perform a single scoring run for parallel execution."""
    seq, confidence_values, scoring_function, query_length, mutation_rate = args
    prob_distributions = calculate_probability_distribution(seq, confidence_values)
    majority_sequence = calculate_majority_sequence(prob_distributions)
    db = build_kmer_database(seq, 10)
    query = sample_from_probabilistic_genome(prob_distributions, query_length)
    query = mutate_sequence(query, mutation_rate)
    query_kmers = create_wmer(query, 10)
    match = identify_hits(db, query_kmers)
    ungapped_results = ungapped_extension(match, query, seq, prob_distributions, 20)
    gapped_results = call_alignment(query, prob_distributions, ungapped_results, scoring_function, majority_sequence)
    filtered_sorted_alignments = filter_and_sort_alignments(gapped_results, 5)
    f1 = 0.0
    if not filtered_sorted_alignments:
        print(f"No alignments found for Query Length: {query_length}, Mutation Rate: {mutation_rate}")
    else:
        best_alignment = filtered_sorted_alignments[0][1]
        aligned_query = best_alignment["query"]
        majority_subsequence = best_alignment["reference"]
        _, _, f1 = compute_metrics(aligned_query, majority_subsequence)

    return f1

def evaluate_scoring_extended(seq, confidence_values, scoring_functions, runs=10, query_lengths=[50, 200, 1000], mutation_rates=[0, 0.15, 0.3, 0.5]):
    results = {scoring_function.__name__: [] for scoring_function in scoring_functions}
    with ProcessPoolExecutor() as executor:
        for scoring_function in scoring_functions:
            print(f"\nEvaluating Scoring Function: {scoring_function.__name__}")
            for query_length in query_lengths:
                for mutation_rate in mutation_rates:
                    args = [(seq, confidence_values, scoring_function, query_length, mutation_rate) for _ in range(runs)]
                    f1_scores = list(executor.map(evaluate_single_run, args))

                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    results[scoring_function.__name__].append((query_length, mutation_rate, mean_f1, std_f1))
                    print(f"Query Length: {query_length}, Mutation Rate: {mutation_rate}, Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

    plot_results(results, query_lengths, mutation_rates)

def plot_results(results, query_lengths, mutation_rates):
    """
    Plot mean F1 scores for different query lengths and scoring functions.
    Each mutation rate gets its own plot, and all scoring functions are shown on the same graph.
    """
    for mutation_rate in mutation_rates:
        plt.figure(figsize=(10, 6))
        for scoring_function, data in results.items():
            means = [mean_f1 for q_len, m_rate, mean_f1, _ in data if m_rate == mutation_rate]
            stds = [std_f1 for q_len, m_rate, _, std_f1 in data if m_rate == mutation_rate]

            plt.errorbar(query_lengths, means, yerr=stds, label=f"{scoring_function}", fmt='-o')

            for i, (q_len, mean) in enumerate(zip(query_lengths, means)):
                plt.annotate(f"{mean:.3f}", (query_lengths[i], mean), textcoords="offset points", xytext=(0, 5), ha='center')

        plt.title(f"F1 Score vs Query Length (Mutation Rate: {mutation_rate})")
        plt.xlabel("Query Length")
        plt.ylabel("Mean F1 Score")
        plt.legend(title="Scoring Function")
        plt.grid(True)
        plt.show()

def main():
    with open("genomesequence.fa", "r") as f_file:
        seq = f_file.readline().strip()

    with open("chr22.maf.ancestors.42000000.complete.boreo.conf.txt", "r") as p_file:
        confidence_values = list(map(float, p_file.readline().split()))

    # Step 2: Calculate Probabilistic Genome
    print("\nCalculating probabilistic genome and majority sequence:")
    prob_distributions = calculate_probability_distribution(seq, confidence_values)
    majority_sequence = calculate_majority_sequence(prob_distributions)

    # Step 3: Build Database
    print("\nBuilding k-mer database:")
    db = build_kmer_database(seq, 11)

    # Step 4: Generate and Process Query
    print("\nGenerate Query:")
    query_length = 100
    query = sample_from_probabilistic_genome(prob_distributions, query_length)
    print(f"Query Sequence: {query}")
    query_kmers = create_wmer(query, 11)

    # Step 5: Identify Hits
    print("\nFinding hits between query and database:")
    match = identify_hits(db, query_kmers)
    print(match)

    # Step 6: Perform Ungapped Extension
    print("\nPerforming Ungapped Extension:")
    delta = 10
    ungapped_results = ungapped_extension(match, query, seq, prob_distributions, delta)
    print_results(ungapped_results, threshold=10)

    # Step 7: Perform Gapped Extension
    print("\nPerforming Gapped Extension:")
    gapped_results = call_alignment(query, prob_distributions, ungapped_results, match_score1, majority_sequence)

    # Step 8: Filter and Sort Alignments
    threshold = 15
    filtered_sorted_alignments = filter_and_sort_alignments(gapped_results, threshold)

    # Step 9: Output Results
    print(f"\nGapped Extension Results (Score >= {threshold}):")
    for (start, end), alignment_data in filtered_sorted_alignments:
        print(f"Start: {alignment_data['ref_start']}, End: {alignment_data['ref_end']}, "
              f"Score: {alignment_data['score']}\nQuery Alignment: {alignment_data['query']}\n"
              f"Reference Alignment: {alignment_data['reference']}\n{'-' * 60}")

if __name__ == "__main__":
    main()
