<<<<<<< HEAD
# Cardioprediction
### cardiodisease prediction with seq model

### indicator
##### -epoch
##### -batch_size
##### -number of layer
##### -number of node
##### -layer and node combination 

##### 5/30 find adquate batch size empirically, batch_size = 200~500
##### there is no mathically accurate answer to adjusting number of layer,
##### but one of stackoverflow answer said "normally number of layer '=.(number of dataset) / (adjusting number)*(input dim + output)
##### 5/30 Adjusting number of layer 3 to 100, but val_acc was 0.5008 or 0.4992. it does not work
##### Adjusting number of layer 100 to 10, val_acc was 73.6% on model evaluate. it was highest percent I've ever exprienced before.
=======
# Cardiovascular disease prediction with DNN
##### dataset : https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
##### dataset detail : 심장병을 가진 환자와 가지지 않은 환자의 다양한 정보(성별, 나이, 콜레스테롤 수치 등)
##### highest val_acc = 73.5%
##### 프로젝트를 진행하면서 깨우친 점:
##### 1. DNN 구성, 전처리 등 DNN(tf,keras)에 대한 아주 기본적인 지식
##### 2. Hyperparameter 수정, 최적의 epoch, batch size, dropout rate 찾기(경험적으로)
##### 2. 현재 dataset 에서 정도의 양질의 data는 얻기 힘들다. only 캐글에서만 얻을 수 있고, 실제 프로젝트를 진행한다면 양질의 labeling된 data부터 구성하는 것이 우선
### Limits
##### Need more data for DNN. there is important data for diagnosis MI. for example, Cardiac marker(Myoglobin, CK-MB, Troponin).
##### If blood Analysis result exist, val_acc will be more higher.
>>>>>>> 33e23e03e9350d2e91cce46dedae36bc53f31e89
