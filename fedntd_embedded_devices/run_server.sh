ROUND=10
MIN_NUM_CLIENTS=4
MIN_SAMPLE_SIZE=2
BATCH_SIZE=16


python server.py \\
--server_address 192.168.0.13:8080 \\
--rounds 10 \\
--min_num_clients 4 \\
--minsample_size 2 \\
--sample_fraction 0.5 \\
--model ResNet18 \\
--num_workers 0 \\
--batch_size 16 \\
--log_host ./log