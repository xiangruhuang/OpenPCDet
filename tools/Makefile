train_and_eval:
	bash scripts/dist_train.sh 8 --cfg_file $(cfg)
	bash scripts/dist_test.sh 8 --cfg_file $(cfg) --batch_size 8 --eval_all

eval:
	bash scripts/dist_test.sh 8 --cfg_file $(cfg) --batch_size 8 --eval_all
