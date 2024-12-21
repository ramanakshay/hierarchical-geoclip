"""
Overall average number of candidates processed: 5388.888888888889
Range of candidates processed: 500.0 to 20000.0
"""

def process_clustering_file(file_path):
    # Initialize variables
    segments = []
    current_segment = []

    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            if 'sigma_index=0' in line:
                # Start of a new segment, save the previous segment if it exists
                if current_segment:
                    segments.append(current_segment)
                current_segment = []
            else:
                # Extract the number from the line
                num_str = line.split('|locations|=')[1].split(' sigma_index=')[0]
                current_segment.append(int(num_str))

        # Save the last segment
        if current_segment:
            segments.append(current_segment)

    # Calculate segment averages
    segment_averages = [sum(segment) / len(segment) for segment in segments]

    # Calculate overall average and range
    overall_average = sum(segment_averages) / len(segment_averages)
    range_of_candidates = (min(segment_averages), max(segment_averages))

    return segment_averages, overall_average, range_of_candidates

file_path = 'clustering_output/out.txt'  # Replace with your actual file path
segment_averages, overall_average, range_of_candidates = process_clustering_file(file_path)

print("Average number of candidates per segment:")
for i, avg in enumerate(segment_averages):
    print(f"Segment {i+1}: {avg}")

print(f"\nOverall average number of candidates processed: {overall_average}")
print(f"Range of candidates processed: {range_of_candidates[0]} to {range_of_candidates[1]}")




