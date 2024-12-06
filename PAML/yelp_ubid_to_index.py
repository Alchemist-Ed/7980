import pandas as pd
import json

# Specify the input JSON file and output .dat file
user_input_file = '/Users/lujiewen/Desktop/papers/datasets/yelp_json/yelp_academic_dataset_user.json'
business_input_file = '/Users/lujiewen/Desktop/papers/datasets/yelp_json/yelp_academic_dataset_business.json'
review_input_file = '/Users/lujiewen/Desktop/papers/datasets/yelp_json/yelp_academic_dataset_review.json'

user_index_mapping_file = '/Users/lujiewen/Desktop/papers/datasets/yelp_json/user_id_to_index.json'
business_index_mapping_file = '/Users/lujiewen/Desktop/papers/datasets/yelp_json/business_id_to_index.json'

user_id_to_index = {}
business_id_to_index = {}
user_current_index = 1  # Start indexing from 1
business_current_index = 1
# extracted_data = []

with open(user_input_file, 'r') as file:
    for line in file:
        entry = json.loads(line.strip())  # Parse each JSON object
        
        # Map user_id to a unique index
        user_id = entry['user_id']
        if user_id not in user_id_to_index:
            user_id_to_index[user_id] = user_current_index
            user_current_index += 1
            
        # friends = entry['friends']
        # friends_list = [friend.strip() for friend in friends.split(',')] 
        # friends_index_id = []
        # friends_index = 1987931
        # for friend in friends_list:
        #     if friend not in user_id_to_index:
        #         user_id_to_index[friend] = friends_index
        #         friends_index += 1
            
            
more_user_index = 1987899    
with open(review_input_file, 'r') as file:
    for line in file:
        entry = json.loads(line.strip())  # Parse each JSON object
        # Map user_id to a unique index
        user_id = entry['user_id']
        if user_id not in user_id_to_index:
            user_id_to_index[user_id] = more_user_index
            more_user_index += 1 
            
friends_index = 1987931
with open(user_input_file, 'r') as file:
    for line in file:
        entry = json.loads(line.strip())  # Parse each JSON object   
        friends = entry['friends']
        friends_list = [friend.strip() for friend in friends.split(',')] 
        for friend in friends_list:
            if friend not in user_id_to_index:
                user_id_to_index[friend] = friends_index
                friends_index += 1

          
with open(business_input_file, 'r') as file:
    for line in file:
        entry = json.loads(line.strip()) 
        
        # Map business_id to a unique index
        business_id = entry['business_id']
        if business_id not in business_id_to_index:
            business_id_to_index[business_id] = business_current_index
            business_current_index += 1        

# save results as json files       
with open(user_index_mapping_file, 'w') as file:
    # json.dump(user_id_to_index, file)
    # for user_id, index in user_id_to_index.items():
    #     json.dump({user_id,index}, file)
    #     # json.dump({user_id:index}, file)
    #     file.write('\n')  # Write a newline after each JSON object
     json.dump(user_id_to_index, file, indent=4)
        
with open(business_index_mapping_file, 'w') as file:
    # json.dump(business_id_to_index, file)
     json.dump(business_id_to_index, file, indent=4)
        

