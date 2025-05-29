This repository contains the implementation of the paper "Pre-training Time Series Models with Stock Data Customization."

## 1. Data Preparation
We use the stock selection benchmark dataset from https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master.
To prepare the data: 
1. Download the dataset and place it in the ./data directory. 
2. Run the following command to process the data:

```
python data.py
```


## 2. Run experiments

The workflow consists of two steps: **Customized Pre-training** and **Fine-tuning for Stock Selection**.

### **Important Parameters**

1. feature_describe: Choose from ['all', 'close_only'] to specify input features.
2. ongoing_task: A list specifying the pre-training tasks. Options include: \
'stock' — stock code classification \
'sector' — stock sector classification \
'mask_avg_price' — masked average prediction \
For example: ['stock'] runs only stock code classification. ['stock', 'sector'] runs combined tasks.
3. pretrain_coef: A hyphen-separated string indicating loss coefficients for the three tasks in the order: [stock, sector, mask_avg_price]. For example, '1-0-0' enables only stock classification. \
Note that tasks that are not include in 'ongoing task' must be arranged with coefficient 0.

### Example: Joint Pre-training and Fine-tuning

```
python run_task.py \
--market_name 'NASDAQ' \
--days 32 \
--feature_describe 'all' \
--ongoing_task 'stock' \
--lr 0.001 \
--pretrain_epoch 100 \
--epoch 0 \
--pretrain_coef '1-0-0' \
--save_pretrain 1
--epoch 50
--loss_alpha 10 
```

### Example: Pre-training Only
```
python run_task.py \
--market_name 'NASDAQ' \
--days 32 \
--feature_describe 'all' \
--ongoing_task 'stock' \
--lr 0.001 \
--pretrain_epoch 100 \
--epoch 0 \
--pretrain_coef '1-0-0' \
--save_pretrain 1
```

### Example: Fine-tuning Only (Using Pre-trained Model)
```
python run_task.py \
--market_name 'NASDAQ' \
--days 32 --feature_describe 'all' \
--ongoing_task 'stock' \
--lr 0.001 \
--loss_alpha 10 \
--pretrain_epoc 0  \
--epoch 50  \
--load_path '[saved_model_path]' \
```

Note: When running fine-tuning separately, make sure parameters such as days and ongoing_task match those used in the pre-training stage.

