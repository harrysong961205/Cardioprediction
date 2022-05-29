# Cardioprediction
### cardiodisease prediction with seq model

##### indicator
##### epoch
##### batch_size
##### number of layer
##### number of node
##### layer and node combination 

##### 5/30 find adquate batch size empirically, batch_size = 200~500
##### there is no mathically accurate answer to adjusting number of layer,
##### but one of stackoverflow answer said "normally number of layer '=.(number of dataset) / (adjusting number)*(input dim + output)
##### 5/30 Adjusting number of layer 3 to 100, but val_acc was 0.5008 or 0.4992. it does not work
##### Adjusting number of layer 100 to 10, val_acc was 73.6% on model evaluate. it was highest percent I've ever exprienced before.
