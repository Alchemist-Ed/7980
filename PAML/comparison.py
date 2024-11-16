import json
 



def extract_and_combine(file1, file2, output_file):
    
    user_data_1 = {}
    with open(file1, 'r') as f1:
        for line in f1:
            record = json.loads(line.strip())
            user_id = record["user_id"]
            user_data_1[user_id] = record

    with open(file2, 'r') as f2, open(output_file, 'w') as output:
        for line in f2:
            record = json.loads(line.strip())
            user_id = record["user_id"]
        
            if user_id in user_data_1:
                output.write("ProposedModel: " + json.dumps(user_data_1[user_id]) + '\n')
                output.write("BaseModel: " + json.dumps(record) + '\n')
                output.write('\n')  # Add a blank line for separation

    # print(f"Combined records saved to {output_file}")

file1_path = 'res/dbook/scores_res/user_scores_epoch_169.txt'
file2_path = 'res/dbook/scores_res/base_user_results_epoch_169.text'
# output_file_path = 'common_users.json'

output_file_path = 'res/dbook/scores_res/common_users.txt'

extract_and_combine(file1_path, file2_path, output_file_path)
