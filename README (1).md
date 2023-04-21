# **README for Fish Counting By Segmentation**

## Table of Contents :

---

```
1. Directory Structure.
2. Directory Information.
3. Dataset Format Required for Training.
4. Setup Environment.
5. Steps for Model Training.
6. Steps for Model Testing.
```

---

## 1. Directory Structure :

```
| -- yolact_worker/
| -- | -- Dockerfile/
| -- | -- | -- yolact_worker.yml
| -- | -- data/
| -- | -- | -- coco.py
| -- | -- | -- config.py
| -- | -- | -- fish/images_100/train.json
| -- | -- | -- fish/images_100/val.json
| -- | -- | -- fish/images_100/
| -- | -- layers
| -- | -- weights/
| -- | -- utils/
| -- | -- backbone.py
| -- | -- yolact.py
| -- | -- train.py
| -- | -- eval.py
| -- | -- yolact-env.yml
| -- | -- build.sh
| -- | -- eval_image.py
| -- | -- docker-compose.yml
| -- | -- yolact-worker
```

---

## 2. Directory Information :

- > `data/ `: Directory contain dataset information like (images on which model to be trained).

- > `data/config.py ` : config file which contain information about model training (like classes,location of dataset)

- > `weights/ ` : Default directory to store tained model file.
- > `train.py ` : Script to train model.
- > `eval.py ` : Script to evaluate trained model.

---

## 3. Dataset Format Required for Training:

We have used COCO dataset format fot the Fish Counting using semantic segmentation project. Although This project also support PASCAL dataset for model training.
How we annotate and export COCO dataset using CVAT:

1. Create New Project and add labels for the classes.
2. Create New Task in the project and import images which we want to annotate.
3. Once images imported succesfully, then open the task.
4. Mark each object in image using Polygon tool and label them accordingly.
5. Once annotation of all images completed, we have to export dataset as per model requirements:

6. Click on **Projects** -> **More** - > **Export Dataset**
7. Select **Export Format** as **_COCO 1.0_**.
8. Check the **Save images**, set name of dataset and then Press _OK_ button. Once download completed we can use this dataset for training.

---

## 4. Setup Environment:

1. > `git clone https://gitlab.com/futurestandard/algorithms/fish_counting_by_segmentation.git`
2. > `cd fish_counting_by_segmentation/yolact_worker`
3. > `conda env create -f fish_count_yolact.yml`
4. > `conda activate fish_count_yolact`
5. > `pip install funcy`
6. > `pip install scikit-multilearn`
7. > `pip install sklearn`
   > Note : User must have miniconda installed on the machine.

---

## 5. Steps for Model Training

Note : These are steps we followed to train model on Fish Counting using semantic Segmentation which have three classes to detect (Sweetfish, Salmon, others)

1. Place the `dataset` in `data/fish/` directory.
2. Divide the images into two parts train & val with ratio of 80 & 20 respectively.
3. Similarly divide the json info with ratio of 80% for train.json & 20% for val.json.
4. Final directory Structure should be like as:

   ```
   | -- yolact_worker/
   | -- | -- data/
   | -- | -- | -- coco.py
   | -- | -- | -- config.py
   | -- | -- | -- fish/images_100/train.json
   | -- | -- | -- fish/images_100/val.json
   | -- | -- | -- fish/images_100/
   ```

5. Divide the COCO JSON file into two separate JSON file(train.json and val.json):

```
   cd utils/
   python cocosplit.py --having-annotations --multi-class -s 0.8 ../path/of/COCO-JSON-file train.json val.json
```

where:

<span style="color: green"> **-s** : Project Split ratio (0.8 means 80% for train.json and 20% for val.json) </span>

<span style="color: green"> **../path/of/COCO-JSON-file** </span>: input COCO-JSON-file

<span style="color: green"> **train.json** </span>: train.json file

<span style="color: green"> **val.json** </span>: val.json file

<span style="color: red"> **Note** : Once script successfully executed, train.json file and val.json will be saved in
./utils directory. You have to place these files in destination directory.</span>

6. Update/add following code snippet in data/config.py :

   #### <span style="color: blue">6.1 :</span> Find "**DATASETS**" section and add following code snippet for **_Dataset Configuration_**

   ```
   fish_dataset = dataset_base.copy({
       'name': 'Fishery Dataset',
       'train_info': 'data/fish/images_100/train.json',
       'train_images': 'data/fish/images_100/',
       'valid_info': 'data/fish/images_100/val.json',
       'valid_images': 'data/fish/images_100/',
       'class_names': ('Sweetfish', 'Salmon', 'Others'),
       'label_map': {1:  1, 2:  2,  3:  3}
   })
   ```

   where:

   <span style="color: green"> name : Project Name </span>

   <span style="color: green"> **train_info** </span>: path of train.json file

   <span style="color: green"> **valid_info** </span>: path of val.json file

   <span style="color: green"> **train_images** </span>: path of images

   <span style="color: green"> **class_names** </span>: Number of classes on which model to be trained

   <span style="color: green"> **label_map** </span>: label for each box

   <span style="color: red"> **Note** : Change the path of dataset Accordingly.</span>

   #### <span style="color: blue">6.2 :</span> Find "**YOLACT v1.0 CONFIGS**" section and following code snippet for **_Model Configuration_**

   ```
   resnet50_fishery_config = yolact_resnet50_config.copy({
       'name': 'fish_dataset',
       # Dataset stuff
       'dataset': fish_dataset,
       'num_classes': len(fish_dataset.class_names) + 1,

       # Image Size
       'max_size': 512,
   })
   ```

   where:

   <span style="color: green">name </span>: name of dataset coniguration

   <span style="color: green">dataset </span>: name of dataset_base(fish_dataset)

   <span style="color: green">num_classes </span>: number of classes

   <span style="color: green">max_size </span>: network-size

   <span style="color: yellow"> **Note** : We have used Resnet50 pre-trained model as backbone architecture.</span>

7. Download `Resnet50` pre-trained model file and place into `weights/` directory

   [resnet50-19c8e357.pth](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view "download")

8. Run script `train.py` :

   > `python train.py --help `

   Note : This will return description of arguments

9. If want to continue with default parameters, then run `train.py` :

   > `python train.py --config=resnet50_fishery_config `

   Note :

   - Training will be started and it will take time for its completion.

   - Once training completed `weight-file` will saved in `weights/` directory.

## 6. Steps for Model Testing:

1.  Run the script `eval.py`:

    > `python eval.py --trained_model=/trained_model_file_path.pth --score_threshold=0.15 --top_k=15 --image=<input.png>:<output.png> --config=resnet50_fishery_config`

    Note:

    - This will save output image in `yolact_worker/` directory.

