# Energy_prediction_GRU
Deep Learning : Energy Prediction using GRU

## Requirment
- Pandas
- Pytorch
- numpy
- tensorboard (1.12)
- tensorboardx

## 설명
딥러닝 모델 성능 실험을 위한 코드입니다. 알맞은 데이터 셋이 없으면 사용할 수 없습니다.
LSTM, GRU, LSTM Seq2Seq 모델을 실험할 수 있도록 구현되어 있으며, 다양한 Hyperparameter를 실험해 볼 수 있습니다.
또한, Teacher forcing을 이용하여 학습하는 실험도 가능하도록 구현되어 있습니다.
Tensorboard를 이용하여 학습 상황을 모니터링 할 수 있으며, test 시에도 마찬가지로 확인 할 수 있습니다.

(util폴더에는 Pytorch의 배치를 만드는 sampler를 필요에 맞게 수정한 것입니다.)

