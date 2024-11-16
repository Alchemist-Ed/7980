import json
file1_path = 'res/dbook/scores_res/user_scores_epoch_169.txt'
file2_path = 'res/dbook/scores_res/base_user_results_epoch_169.text'
# output_file_path = 'common_users.json'

output_file_path = 'res/dbook/scores_res/same_user_comparison.txt'

# Load data from file1
with open(file1_path, 'r') as f1:
    user_data_1 = {json.loads(line)["user_id"]: json.loads(line) for line in f1}

# Load data from file2 and compare with file1
with open(file2_path, 'r') as f2, open(output_file_path, 'w') as output:
    for line in f2:
        record = json.loads(line)
        user_id = record["user_id"]

        # Check if user_id exists in file1 and meets conditions
        if user_id in user_data_1:
            file1_record = user_data_1[user_id]
            rmse_file1 = file1_record.get("rmse")
            ndcg_file1 = file1_record.get("ndcg@5")
            rmse_file2 = record.get("rmse")
            ndcg_file2 = record.get("ndcg@5")
            
            # Compare based on rmse and ndcg@5 values
            if rmse_file1 < rmse_file2 and ndcg_file1 > ndcg_file2:
                output.write("ProposedModel: " + json.dumps(file1_record) + '\n')
                output.write("BaseModel: " + json.dumps(record) + '\n')
                output.write('\n')  # Blank line for separation
