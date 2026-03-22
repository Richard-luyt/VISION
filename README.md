# VISION

This is a CNN+RNN based model that predicts the position of the object in sequence $T+1$ with the input from previous $T$ pictures. 

## More info about the model

The model consists of a pre-trained ResNet18 from torchvision, and an LSTM model. The last two layers of the ResNet18 is deleted and replaced with a Adaptive Average Pooling layer which outputs a tensor with dimension `[B*T 512 2 2]`. Then the tensor is flattened and sent to a Linear layer followed by a ReLU and Dropout layer for down sampling. Then the LSTM model with `hidden_size 256` and `num_layers=2` followed by a Linear Layer outputs the prediction. The prediction is a tensor with four elements `[delta x, delta y, delta w, delta h]

## Some results
<img width="1197" height="396" alt="屏幕截图 2026-03-03 104635" src="https://github.com/user-attachments/assets/66b0045c-e6e7-412b-9c21-2c6958758059" />
<img width="1193" height="395" alt="屏幕截图 2026-03-03 104626" src="https://github.com/user-attachments/assets/9c3835ea-e65e-41ce-bc4a-54bd9221190a" />
<img width="1183" height="392" alt="屏幕截图 2026-03-03 104619" src="https://github.com/user-attachments/assets/ecef3e9e-dc32-4714-acc4-8329f75ad9f7" />
