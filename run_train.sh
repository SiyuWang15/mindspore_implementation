# time=$(date "+%m%d-%H-%M-%S")

# # train FC coarse channel estimation
# # use config file in configs/y2h_config_fc.yml
# # The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/FC/default ]

cd train
CUDA_VISIBLE_DEVICES=0 python main.py --runner y2h --time default --run_mode fc -Pn 8

# # train CNN finer channel estimation 
# # use config file in configs/y2h_config_cnn.yml
# # The results and logging are saved in [ ./workspace/ResnetY2HEstimator/mode_0_Pn_8/CNN/default ]

CUDA_VISIBLE_DEVICES=1 python main.py --runner y2h --time default --run_mode cnn -Pn 8