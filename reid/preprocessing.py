import os
import shutil
import random
from tqdm import tqdm

def separate_to_client_equal(base_dir, image_dir, k, query_ratio, loc, sn, max_labels_per_client):
    # Make k client directories
    for i in range(1, k+1):
        client_base_dir = os.path.join(base_dir, f"person{i}")
        query_dir = os.path.join(client_base_dir, 'query')
        gallery_dir = os.path.join(client_base_dir, 'gallery')
        train_dir = os.path.join(client_base_dir, 'train')
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(gallery_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)

    # Dictionary to keep track of assigned clients for each person_id
    person_to_client = {}
    client_label_count = {i: 0 for i in range(1, k+1)}  # Track the number of labels per client

    # Walk through all subdirectories and files in the image directory
    for root, _, files in os.walk(image_dir):
        for file in tqdm(files):
            
            if random.random() < 0.3:
                pass
            else:
                continue
            
            if file.endswith('.png'):  # Adjust the file extension if necessary
                parts = file.split("_")
                
                if loc not in parts[0] or sn not in parts[2]:
                    continue
                person_id = int(parts[1][1:])
                cctv_id = parts[3]
                image_id = parts[-1].split('.')[0]


                # Set new file name
                new_file_name = f"{person_id}_{cctv_id}_{image_id}.jpg"

                # Assign client folder based on person_id
                if person_id not in person_to_client:
                    available_clients = [i for i in range(1, k+1) if client_label_count[i] < max_labels_per_client]
                    if available_clients:  # Assign person_id only if there is an available client
                        client_index = random.choice(available_clients)
                        person_to_client[person_id] = client_index
                        client_label_count[client_index] += 1
                    else:
                        continue  # Skip if all clients have reached max labels

                client_index = person_to_client[person_id]
                client_base_dir = os.path.join(base_dir, f"person{client_index}")

                # Randomly decide between query and gallery based on query_ratio
                if "Training" in image_dir:
                    person_dir = os.path.join(client_base_dir, 'train', f"{person_id}")
                else: # "Validation"
                    if random.random() < query_ratio:
                        person_dir = os.path.join(client_base_dir, 'query', f"{person_id}")
                    else:
                        person_dir = os.path.join(client_base_dir, 'gallery', f"{person_id}")

                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)

                src_file = os.path.join(root, file)
                dst_file = os.path.join(person_dir, new_file_name)
                
                print(src_file, dst_file)
                # shutil.copy(src_file, dst_file)  # Copy image file

    print(f"Dataset has been separated by person_id into {k} clients, with a max of {max_labels_per_client} labels per client.")


if __name__ == '__main__':
    base_dir = '/home/sungwoo/etri'  # Base directory
    image_dir = '/home/sungwoo/etri/person_dataset/Validation'  # Image directory Training or Validation
    k = 5  # Number of clients
    query_ratio = 0.2  # Ratio of query images
    loc = "IN" # OUT or IN
    sn = "SN1" # SN1 or SN2 or SN3 or SN4
    separate_to_client_equal(base_dir, image_dir, k, query_ratio, loc, sn, 10)
