ROUND=10
MIN_NUM_CLIENTS=4
MIN_SAMPLE_SIZE=2
BATCH_SIZE=16


python server.py \\
--server_address 192.168.0.13:8080 \\
--rounds 30 \\
--epochs 1 \\
--lr 0.01 \\
--min_num_clients 4 \\
--minsample_size 4 \\
--sample_fraction 1.0 \\
--model Net \\
--num_workers 0 \\
--batch_size 32 \\
--log_host ./log