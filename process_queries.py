import json
import csv
import os

def process_json_to_csv(json_file, output_csv):
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # Write to CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['ID', 'Question', 'Context', 'Answer', 'Sections', 'Pages'])
        
        # Write data
        for query in queries:
            # Join sections and pages with semicolons if they exist
            sections = ';'.join(query['references']['sections']) if 'references' in query and 'sections' in query['references'] else ''
            pages = ';'.join(query['references']['pages']) if 'references' in query and 'pages' in query['references'] else ''
            
            writer.writerow([
                query['query_id'],
                query['question'],
                query['context'],
                query['answer'],
                sections,
                pages
            ])

def main():
    # Process sample queries
    process_json_to_csv('refer/(NEW) sample_queries.json', 'sample_submission.csv')
    process_json_to_csv('refer/queries.json', 'full_submission.csv')

if __name__ == "__main__":
    main() 